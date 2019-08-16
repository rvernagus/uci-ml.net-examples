open Microsoft.ML
open Microsoft.ML.Data
open System.IO
open System.Net
open Annealing


let shuffle (context : MLContext) dataView =
    context.Data.ShuffleRows(dataView)

let append (chain : EstimatorChain<'T>) transform =
    chain.Append(transform)

let onehot (context : MLContext) (column : string) =
    context.Transforms.Categorical.OneHotEncoding(column)

let concatenate (context : MLContext) outputColumnName inputColumnNames =
    context.Transforms.Concatenate(outputColumnName = outputColumnName, inputColumnNames = inputColumnNames)

let mapValueToKey (context : MLContext) sourceColumn destinationColumn =
    context.Transforms.Conversion.MapValueToKey(inputColumnName = sourceColumn, outputColumnName = destinationColumn)

let printCvResultMetrics (cvResults : TrainCatalogBase.CrossValidationResult<MulticlassClassificationMetrics> seq) =
    do
        printfn "------------------\nCross Validation Metrics\n------------------"
        cvResults
        |> Seq.map (fun cvResult -> cvResult.Metrics.MacroAccuracy)
        |> Seq.average
        |> printfn "Accuracy: %f"; cvResults
        |> Seq.map (fun cvResult -> cvResult.Metrics.LogLoss)
        |> Seq.average
        |> printfn "Log Loss: %f"
    
    cvResults

let printMetrics (metrics : MulticlassClassificationMetrics) =
    do
        printfn "------------------\nTest Metrics\n------------------"
        printfn "Accuracy: %f" metrics.MacroAccuracy
        printfn "Log Loss: %f" metrics.LogLoss
        printfn "Confusion Matrix:"
        printfn "%s" <| metrics.ConfusionMatrix.GetFormattedConfusionTable()

let downcastEstimator (e : IEstimator<'a>) =
    match e with
    | :? IEstimator<ITransformer> as p -> p
    | _ -> failwith "The estimator has to be an instance of IEstimator<ITransformer>."

let makeEstimator (context : MLContext) featureColumnName =
    context.MulticlassClassification.Trainers.LbfgsMaximumEntropy(featureColumnName = featureColumnName)
    |> downcastEstimator

let transform (transformer : ITransformer) dataView =
    transformer.Transform(dataView)

let crossValidate (context : MLContext) estimator numberOfFolds dataView =
    context.MulticlassClassification.CrossValidate(dataView, estimator, numberOfFolds = numberOfFolds)


[<EntryPoint>]
let main argv =
    if not <| File.Exists("anneal.data") then
        use client = new WebClient()
        client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/annealing/anneal.data", "anneal.data")

    if not <| File.Exists("anneal.test") then
        use client = new WebClient()
        client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/annealing/anneal.test", "anneal.test")

    let context = new MLContext()

    let trainDataView =
        context.Data.LoadFromTextFile<AnnealingData>("anneal.data", hasHeader = false, separatorChar = ',')
        |> shuffle context

    let testDataView =
        context.Data.LoadFromTextFile<AnnealingData>("anneal.test", hasHeader = false, separatorChar = ',')
        |> shuffle context

    let featureColumns =
        [|
            "Family"; "ProductType"; "Steel"; "Carbon"; "Hardness"; "TemperRolling"; "Condition"; "Formability"; "Strength";
            "NonAgeing"; "SurfaceFinish"; "SurfaceQuality"; "Enamelability"; "Bc"; "Bf"; "Bt"; "BwMe"; "Bl"; "M"; "Chrom";
            "Phos"; "Cbond"; "Marvi"; "Exptl"; "Ferro"; "Corr"; "BlueBrightVarnClean"; "Lustre"; "Jurofm"; "S"; "P"; "Shape";
            "Thick"; "Width"; "Len"; "Oil"; "Bore"; "Packing"
        |]

    let categoricalColumns =
        [|
            "Family"; "ProductType"; "Steel"; "TemperRolling"; "Condition"; "Formability"; "NonAgeing"; "SurfaceFinish";
            "SurfaceQuality"; "Enamelability"; "Bc"; "Bf"; "Bt"; "BwMe"; "Bl"; "M"; "Chrom"; "Phos"; "Cbond"; "Marvi";
            "Exptl"; "Ferro"; "Corr"; "BlueBrightVarnClean"; "Lustre"; "Jurofm"; "S"; "P"; "Shape"; "Oil"; "Bore"; "Packing"
        |]

    let transformer =
        categoricalColumns
        |> Seq.map (onehot context)
        |> Seq.fold append (EstimatorChain())
        |> append <| mapValueToKey context "Label" "Label"
        |> append <| concatenate context "Features" featureColumns
        |> fun pipeline -> pipeline.Fit(trainDataView)

    let estimator = makeEstimator context "Features"

    let model =
        trainDataView
        |> transform transformer
        |> crossValidate context estimator 3
        |> Seq.maxBy (fun cvResult -> cvResult.Metrics.MacroAccuracy)
        |> fun cvResult -> cvResult.Model
    
    do
        trainDataView
        |> transform transformer
        |> crossValidate context estimator 3
        |> Seq.maxBy (fun cvResult -> cvResult.Metrics.MacroAccuracy)
        |> fun cvResult -> cvResult.Model
        |> transform <| transform transformer testDataView
        |> context.MulticlassClassification.Evaluate
        |> printMetrics

    // Print some individual predictions
    let testRecords =
        testDataView
        |> context.Data.ShuffleRows
        |> fun data -> context.Data.TakeRows(data, 5L)

    let actualPrices =
        testRecords
        |> transform transformer
        |> fun dv -> dv.Preview().RowView
        |> Seq.map (fun record -> record.Values.[103])

    let predictedPrices =
        testRecords
        |> transform transformer
        |> transform model
        |> fun resultView -> resultView.Preview().RowView
        |> Seq.map (fun record -> record.Values.[105])

    do
        printfn "------------------"
        Seq.zip actualPrices predictedPrices
        |> Seq.iter (printfn "%A")

    0
