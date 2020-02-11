open Microsoft.ML
open Microsoft.ML.Data
open System.IO
open System.Net
open Annealing
open FunctionalMl


[<EntryPoint>]
let main argv =
    if not <| File.Exists("anneal.data") then
        use client = new WebClient()
        client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/annealing/anneal.data", "anneal.data")

    if not <| File.Exists("anneal.test") then
        use client = new WebClient()
        client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/annealing/anneal.test", "anneal.test")

    let ml = new MlWrapper()

    let trainData =
        ml.Context.Data.LoadFromTextFile<AnnealingData>("anneal.data", hasHeader = false, separatorChar = ',')
        |> ml.Shuffle

    let testData =
        ml.Context.Data.LoadFromTextFile<AnnealingData>("anneal.test", hasHeader = false, separatorChar = ',')
        |> ml.Shuffle

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

    let pipeline =
        categoricalColumns
        |> Seq.map ml.Onehot // Create a one-hot encoder for each categorical column
        |> Seq.fold ml.Append (EstimatorChain()) // Add the encoders to a new EstimatorChain
        |> ml.Append <| ml.MapValueToKey "Label" "Label" // Map labels keys
        |> ml.Append <| ml.Concatenate "Features" featureColumns // Concatenate feature columns into a single new column
        |> ml.Append <| ml.MapKeyToValue "Label" "LabelValue"

    let transformer =
        pipeline
        |> ml.Fit trainData // Fit our pipeline on the training data

    // Print transformed data
    do
        let transformedData =
            trainData
            |> ml.Transform transformer

        printfn "------------------\nData As Loaded\n------------------"
        ml.Context.Data.CreateEnumerable<AnnealingData>(trainData, reuseRowObject = false)
        |> Seq.take 3
        |> Seq.iter (printfn "%A")

        let test = ml.Context.Data.CreateEnumerable<AnnealingDataTransformed>(transformedData, reuseRowObject = false)

        printfn "------------------\nTransformed Data\n------------------"
        ml.Context.Data.CreateEnumerable<AnnealingDataTransformed>(transformedData, reuseRowObject = false)
        |> Seq.take 3
        |> Seq.iter (printfn "%A")

    let estimator =
        ml.Context.MulticlassClassification.Trainers.LbfgsMaximumEntropy(featureColumnName = "Features")
        |> ml.DowncastEstimator

    let model =
        trainData // Begin with the training data
        |> ml.Transform transformer // Transform using the transformer built above
        |> ml.CrossValidateMulticlassClassification estimator 3 // 3-fold cross-validation
        |> ml.PrintMulticlassClassificationCvMetrics // Print cross-fold metrics
        |> Seq.maxBy (fun cvResult -> cvResult.Metrics.MacroAccuracy) // Select the best model by Accuracy
        |> fun cvResult -> cvResult.Model
    
    do
        model
        |> ml.Transform <| ml.Transform transformer testData // Transform the test data and get predictions
        |> ml.Context.MulticlassClassification.Evaluate // Get test metrics
        |> ml.PrintMulticlassClassificationMetrics

    // Show some sample predictions
    let sampleData =
        testData
        |> ml.Transform transformer
        |> ml.Transform model

    let postPredictionPipeline =
        EstimatorChain()
        |> ml.Append <| ml.MapKeyToValue "PredictedLabel" "PredictedLabelValue"
        |> ml.Append <| ml.MapKeyToValue "Label" "LabelValue"
        |> ml.Fit sampleData

    let samplePredictions =
        sampleData
        |> ml.Transform postPredictionPipeline

    do
        printfn "------------------\nSample Predictions\n------------------"
        ml.Context.Data.CreateEnumerable<AnnealingPrediction>(samplePredictions, reuseRowObject = false)
        |> Seq.take 5
        |> Seq.iter (printfn "%A")

    0
