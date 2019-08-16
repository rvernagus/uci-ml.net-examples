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

let pca (context : MLContext) inputColumnName outputColumnName =
    context.Transforms.ProjectToPrincipalComponents(outputColumnName = outputColumnName, inputColumnName = inputColumnName, rank = 15, ensureZeroMean = true)

let downcastEstimator (e : IEstimator<'a>) =
    match e with
    | :? IEstimator<ITransformer> as p -> p
    | _ -> failwith "The estimator has to be an instance of IEstimator<ITransformer>."

let makeEstimator (context : MLContext) featureColumnName =
    context.MulticlassClassification.Trainers.LbfgsMaximumEntropy(featureColumnName = "FeaturesPCA")
    |> downcastEstimator

let transform (transformer : ITransformer) dataView =
    transformer.Transform(dataView)

let crossValidate (context : MLContext) estimator numberOfFolds dataView =
    context.MulticlassClassification.CrossValidate(dataView, estimator, numberOfFolds = numberOfFolds)

let printMetrics (metrics : MulticlassClassificationMetrics) =
    for c in metrics.ConfusionMatrix.Counts do
        printfn "%A" c
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

    // TODO: group by label to get counts over test data
    let cursor = testDataView.GetRowCursor(testDataView.Schema)
    let labels = new System.Collections.Generic.List<int32>()
    while cursor.MoveNext() do
        let getter = cursor.GetGetter<int32>(testDataView.Schema.["Label"])
        let mutable value = int32 0
        getter.Invoke(&value)
        labels.Add(value)
    printfn "%A" labels
    let labelCounts =
        labels
        |> Seq.groupBy id
        |> Map.ofSeq
        |> Map.map (fun _ labels -> labels |> Seq.length)
    printfn "%A" labelCounts

    let transformer =
        EstimatorChain()
        |> append <| mapValueToKey context "Label" "Label"
        |> append <| concatenate context "Features" featureColumns
        //|> append <| replaceMissing context "Features" "Features"
        |> append <| normalize context "Features" "FeaturesNorm"
        |> append <| pca context "FeaturesNorm" "FeaturesPCA"
        |> (fun pipeline -> pipeline.Fit(trainDataView))

    let transformedTrainData = transformer.Transform(trainDataView)
    let transformedTestData = transformer.Transform(testDataView)

    //transformedTrainData.Preview().RowView
    //|> Seq.take 5
    //|> Seq.map (fun row -> row.Values.[284])
    //|> Seq.map (fun v -> v.Value :?> VBuffer<single>)
    //|> Seq.map (fun vec -> vec.DenseValues())
    //|> Seq.iter (fun vals -> printfn "%A" (Seq.item 13 vals))

    
    let estimator = makeEstimator context "FeaturesPCA"
    
    //let model =
    //    trainDataView
    //    |> transform transformer
    //    |> crossValidate context estimator 3
    //    |> Seq.maxBy (fun cvResult -> cvResult.Metrics.MacroAccuracy)
    //    |> fun cvResult -> cvResult.Model

    //// Print some individual predictions
    //let testRecords =
    //    testDataView
    //    |> context.Data.ShuffleRows
    //    |> fun data -> context.Data.TakeRows(data, 50L)

    //let predictedClasses=
    //    testRecords
    //    |> transform transformer
    //    |> transform model
    //    |> fun resultView -> resultView.Preview().RowView
    //    |> Seq.map (fun record -> record.Values.[285])
    //    |> Seq.iter (printfn "%A")
        //|> Seq.map (fun v -> v.Value :?> VBuffer<single>)
        //|> Seq.map (fun vec -> vec.DenseValues())
        //|> Seq.iter (fun vals -> printfn "%A" (Seq.item 278 vals))



    do
        context.MulticlassClassification.CrossValidate(transformedTrainData, estimator, numberOfFolds = 3)
        |> Seq.minBy (fun cvResult -> cvResult.Metrics.LogLoss)
        |> fun cvResult -> cvResult.Model.Transform(transformedTestData)
        |> context.MulticlassClassification.Evaluate
        |> printMetrics

    0
