open Microsoft.ML
open Microsoft.ML.Data
open Adult
open System.IO
open System.Net
open System.Collections.Generic


let append (chain : EstimatorChain<'T>) transform =
    chain.Append(transform)

let onehot (context : MLContext) (column : string) =
    context.Transforms.Categorical.OneHotEncoding(column)

let concatenate (context : MLContext)  outputColumnName inputColumnNames =
    context.Transforms.Concatenate(outputColumnName = outputColumnName, inputColumnNames = inputColumnNames)

let mapValue (context : MLContext)  outputColumnName (keyValuePairs : IEnumerable<KeyValuePair<string, bool>>) inputColumnName =
    context.Transforms.Conversion.MapValue(outputColumnName, keyValuePairs, inputColumnName)

let bin (context : MLContext)  inputColumn outputColumn =
    context.Transforms.NormalizeBinning(outputColumnName = outputColumn, inputColumnName = inputColumn)

let downcastEstimator (e : IEstimator<'a>) =
    match e with
    | :? IEstimator<ITransformer> as p -> p
    | _ -> failwith "The estimator has to be an instance of IEstimator<ITransformer>."

let printMetrics (metrics : CalibratedBinaryClassificationMetrics) =
    printfn "Accuracy: %f" metrics.Accuracy
    printfn "Log Loss: %f" metrics.LogLoss
    printfn "Area Under ROC Curve: %f" metrics.AreaUnderRocCurve
    printfn "F1 Score: %f" metrics.F1Score


[<EntryPoint>]
let main argv =
    if not <| File.Exists("adult.data") then
        use client = new WebClient()
        client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", "adult.data")

    if not <| File.Exists("adult.test") then
        use client = new WebClient()
        client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", "adult.test")
    
    let context = new MLContext()
    
    let trainDataView =
        context.Data.LoadFromTextFile<AdultData>("adult.data", hasHeader = false, separatorChar = ',')
        |> context.Data.ShuffleRows
    let testDataView =
        context.Data.LoadFromTextFile<AdultData>("adult.test", hasHeader = false, separatorChar = ',')
        |> context.Data.ShuffleRows
    
    let featureColumns = [| "Age"; "WorkClass"; "Fnlwgt"; "Education"; "EducationNum"; "MaritalStatus"; "Occupation"; "Relationship"; "Race"; "Sex"; "CapitalGain"; "CapitalLoss"; "HoursPerWeek"; "NativeCountry" |]
    let categoricalColumns = [| "WorkClass"; "Education"; "MaritalStatus"; "Occupation"; "Relationship"; "Race"; "Sex"; "NativeCountry" |]
    let labelLookup =
        [|
            KeyValuePair("<=50K", false)
            KeyValuePair("<=50K.", false)
            KeyValuePair(">50K", true)
            KeyValuePair(">50K.", true)
        |]

    let pipeline =
        categoricalColumns
        |> Seq.map (onehot context)
        |> Seq.fold append (EstimatorChain())
        |> append <| mapValue context "Label" labelLookup "Label"
        |> append <| concatenate context "Features" featureColumns
        |> append <| bin context "Features" "FeaturesNorm"

    let transformer = pipeline.Fit(trainDataView)
    let transformedTrainData = transformer.Transform(trainDataView)
    let transformedTestData = transformer.Transform(testDataView)
    
    let estimator = context.BinaryClassification.Trainers.SdcaLogisticRegression(featureColumnName = "FeaturesNorm")
    let finalEstimator = downcastEstimator estimator
    
    context.BinaryClassification.CrossValidate(transformedTrainData, finalEstimator, numberOfFolds = 3)
    |> Seq.maxBy (fun cvResult -> cvResult.Metrics.Accuracy)
    |> fun cvResult -> cvResult.Model.Transform(transformedTestData)
    |> context.BinaryClassification.Evaluate
    |> printMetrics

    0
