open Microsoft.ML
open Microsoft.ML.Data
open System.IO
open System.Net
open Annealing


let append (chain : EstimatorChain<'T>) transform =
    chain.Append(transform)

let onehot (context : MLContext) (column : string) =
    context.Transforms.Categorical.OneHotEncoding(column)

let concatenate (context : MLContext) outputColumnName inputColumnNames =
    context.Transforms.Concatenate(outputColumnName = outputColumnName, inputColumnNames = inputColumnNames)

let mapValueToKey (context : MLContext) sourceColumn destinationColumn =
    context.Transforms.Conversion.MapValueToKey(inputColumnName = sourceColumn, outputColumnName = destinationColumn)

let minMax (context : MLContext) inputColumn outputColumn =
    context.Transforms.NormalizeMinMax(outputColumnName = outputColumn, inputColumnName = inputColumn)

let printMetrics (metrics : MulticlassClassificationMetrics) =
    printfn "Accuracy: %f" metrics.MacroAccuracy
    printfn "Log Loss: %f" metrics.LogLoss
    printfn "Confusion Matrix:"
    printfn "%s" <| metrics.ConfusionMatrix.GetFormattedConfusionTable()

let downcastEstimator (e : IEstimator<'a>) =
    match e with
    | :? IEstimator<ITransformer> as p -> p
    | _ -> failwith "The estimator has to be an instance of IEstimator<ITransformer>."


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
        |> context.Data.ShuffleRows

    let testDataView =
        context.Data.LoadFromTextFile<AnnealingData>("anneal.test", hasHeader = false, separatorChar = ',')
        |> context.Data.ShuffleRows

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
        |> append <| minMax context "Features" "FeaturesNorm"
        |> (fun pipeline -> pipeline.Fit(trainDataView))

    let transformedTrainData = transformer.Transform(trainDataView)
    let transformedTestData = transformer.Transform(testDataView)

    let estimator = context.MulticlassClassification.Trainers.LbfgsMaximumEntropy(featureColumnName = "Features")
    let finalEstimator = downcastEstimator estimator

    context.MulticlassClassification.CrossValidate(transformedTrainData, finalEstimator, numberOfFolds = 3)
    |> Seq.maxBy (fun cvResult -> cvResult.Metrics.MacroAccuracy)
    |> fun cvResult -> cvResult.Model.Transform(transformedTestData)
    |> context.MulticlassClassification.Evaluate
    |> printMetrics

    0
