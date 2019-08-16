open Microsoft.ML
open Microsoft.ML.Data
open Adult
open System.IO
open System.Net
open System.Collections.Generic


let printCvResultMetrics (cvResults : TrainCatalogBase.CrossValidationResult<CalibratedBinaryClassificationMetrics> seq) =
    do
        printfn "------------------\nCross Validation Metrics\n------------------"
        cvResults
        |> Seq.map (fun cvResult -> cvResult.Metrics.Accuracy)
        |> Seq.average
        |> printfn "Accuracy: %f"; cvResults
        |> Seq.map (fun cvResult -> cvResult.Metrics.AreaUnderRocCurve)
        |> Seq.average
        |> printfn "Area Under Roc Curve: %f"; cvResults
        |> Seq.map (fun cvResult -> cvResult.Metrics.F1Score)
        |> Seq.average
        |> printfn "F1 Score: %f"

    cvResults

let loadAndShuffle (context : MLContext) filePath =
    context.Data.LoadFromTextFile<AdultData>(filePath, hasHeader = false, separatorChar = ',')
    |> context.Data.ShuffleRows

let append (chain : EstimatorChain<'T>) transform =
    chain.Append(transform)

let onehot (context : MLContext) (column : string) =
    context.Transforms.Categorical.OneHotEncoding(column)

let concatenate (context : MLContext)  outputColumnName inputColumnNames =
    context.Transforms.Concatenate(outputColumnName = outputColumnName, inputColumnNames = inputColumnNames)

let mapValue (context : MLContext)  outputColumnName (keyValuePairs : IEnumerable<KeyValuePair<string, bool>>) inputColumnName =
    context.Transforms.Conversion.MapValue(outputColumnName, keyValuePairs, inputColumnName)

let normalize (context : MLContext)  inputColumn outputColumn =
    context.Transforms.NormalizeBinning(outputColumnName = outputColumn, inputColumnName = inputColumn)

let downcastEstimator (e : IEstimator<'a>) =
    match e with
    | :? IEstimator<ITransformer> as p -> p
    | _ -> failwith "The estimator has to be an instance of IEstimator<ITransformer>."

let makeEstimator (context : MLContext) featureColumnName =
    context.BinaryClassification.Trainers.SdcaLogisticRegression(featureColumnName = featureColumnName)
    |> downcastEstimator

let transform (transformer : ITransformer) dataView =
    transformer.Transform(dataView)

let crossValidate (context : MLContext) estimator numberOfFolds dataView =
    context.BinaryClassification.CrossValidate(dataView, estimator, numberOfFolds = numberOfFolds)

let printMetrics (metrics : CalibratedBinaryClassificationMetrics) =
    do
        printfn "------------------\nTest Metrics\n------------------"
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
    
    let trainDataView = loadAndShuffle context "adult.data"
    let testDataView = loadAndShuffle context "adult.test"
    
    let featureColumns = [| "Age"; "WorkClass"; "Fnlwgt"; "Education"; "EducationNum"; "MaritalStatus"; "Occupation"; "Relationship"; "Race"; "Sex"; "CapitalGain"; "CapitalLoss"; "HoursPerWeek"; "NativeCountry" |]
    let categoricalColumns = [| "WorkClass"; "Education"; "MaritalStatus"; "Occupation"; "Relationship"; "Race"; "Sex"; "NativeCountry" |]
    let labelLookup =
        [|
            KeyValuePair("<=50K", false)
            KeyValuePair("<=50K.", false)
            KeyValuePair(">50K", true)
            KeyValuePair(">50K.", true)
        |]

    let transformer =
        categoricalColumns
        |> Seq.map (onehot context) // Create a one-hot encoder for each categorical column
        |> Seq.fold append (EstimatorChain()) // Add the encoders to a new EstimatorChain
        |> append <| mapValue context "Label" labelLookup "Label" // Map labels to either true or false
        |> append <| concatenate context "Features" featureColumns // Concatenate feature columns into a single new column
        |> append <| normalize context "Features" "FeaturesNorm" // Normalize features into a new column, FeaturesNorm
        |> (fun pipeline -> pipeline.Fit(trainDataView)) // Fit our pipeline on the training data
    
    let estimator = makeEstimator context "FeaturesNorm"
    
    do
        trainDataView  // Begin with the training data
        |> transform transformer // Transform using the transformer built above
        |> crossValidate context estimator 3 // 3-fold cross-validation
        |> printCvResultMetrics // Print cross-fold metrics
        |> Seq.maxBy (fun cvResult -> cvResult.Metrics.Accuracy)  // Select the best model by accuracy
        |> fun cvResult -> cvResult.Model
        |> transform <| transform transformer testDataView // Transform the test data and get predictions
        |> context.BinaryClassification.Evaluate // Get test metrics
        |> printMetrics

    0
