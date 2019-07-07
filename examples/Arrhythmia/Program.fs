open System.IO
open System.Net
open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Transforms


let makeSingleColumn index =
    new TextLoader.Column(string index, DataKind.Single, index)

let append (chain : EstimatorChain<'T>) transform =
    chain.Append(transform)

let concatenate (context : MLContext) outputColumnName inputColumnNames =
    context.Transforms.Concatenate(outputColumnName = outputColumnName, inputColumnNames = inputColumnNames)
    
let mapValueToKey (context : MLContext) sourceColumn destinationColumn =
    context.Transforms.Conversion.MapValueToKey(inputColumnName = sourceColumn, outputColumnName = destinationColumn)

let normalize (context : MLContext) inputColumn outputColumn =
    context.Transforms.NormalizeMinMax(outputColumnName = outputColumn, inputColumnName = inputColumn)
   
let replaceMissing (context : MLContext) (inputColumn : string) (outputColumn : string) =
    context.Transforms.ReplaceMissingValues(inputColumn, replacementMode = MissingValueReplacingEstimator.ReplacementMode.Mean)

let downcastEstimator (e : IEstimator<'a>) =
    match e with
    | :? IEstimator<ITransformer> as p -> p
    | _ -> failwith "The estimator has to be an instance of IEstimator<ITransformer>."

let printMetrics (metrics : MulticlassClassificationMetrics) =
    printfn "Accuracy: %f" metrics.MacroAccuracy
    printfn "Log Loss: %f" metrics.LogLoss
    printfn "Confusion Matrix:"
    printfn "%s" <| metrics.ConfusionMatrix.GetFormattedConfusionTable()


[<EntryPoint>]
let main argv =
    if not <| File.Exists("arrhythmia.data") then
        use client = new WebClient()
        client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data", "arrhythmia.data")
    
    let context = new MLContext()

    let columns =
        seq { 0..278 }
        |> Seq.map makeSingleColumn
        |> Seq.append [ new TextLoader.Column("Label", DataKind.Int32, 279) ]
        |> Seq.toArray

    let featureColumns =
        seq { 0..278 }
        |> Seq.map string
        |> Seq.toArray

    let textLoader = context.Data.CreateTextLoader(columns, hasHeader = false, separatorChar = ',')
    
    let allDataView = textLoader.Load("arrhythmia.data")

    let trainDataView, testDataView =
        let shuffledData = context.Data.ShuffleRows(allDataView)
        let split = context.Data.TrainTestSplit(shuffledData, testFraction = 0.2)
        split.TrainSet, split.TestSet

    let transformer =
        EstimatorChain()
        |> append <| mapValueToKey context "Label" "Label"
        |> append <| concatenate context "Features" featureColumns
        |> append <| replaceMissing context "Features" "Features"
        |> append <| normalize context "Features" "FeaturesNorm"
        |> (fun pipeline -> pipeline.Fit(trainDataView))

    let transformedTrainData = transformer.Transform(trainDataView)
    let transformedTestData = transformer.Transform(testDataView)
    
    let estimator = context.MulticlassClassification.Trainers.LbfgsMaximumEntropy(featureColumnName = "FeaturesNorm")
    let finalEstimator = downcastEstimator estimator
    
    context.MulticlassClassification.CrossValidate(transformedTrainData, finalEstimator, numberOfFolds = 3)
    |> Seq.minBy (fun cvResult -> cvResult.Metrics.LogLoss)
    |> fun cvResult -> cvResult.Model.Transform(transformedTestData)
    |> context.MulticlassClassification.Evaluate
    |> printMetrics

    0
