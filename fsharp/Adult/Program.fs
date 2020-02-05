open Microsoft.ML
open Microsoft.ML.Data
open Adult
open System.IO
open System.Net
open System.Collections.Generic
open FunctionalMl


[<EntryPoint>]
let main argv =
    if not <| File.Exists("adult.data") then
        use client = new WebClient()
        client.Proxy <- new WebProxy("http://localhost:3128")
        client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", "adult.data")

    if not <| File.Exists("adult.test") then
        use client = new WebClient()
        client.Proxy <- new WebProxy("http://localhost:3128")
        client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", "adult.test")

    //let context = new MLContext()
    let ml = MlWrapper()

    let trainData =
        ml.Context.Data.LoadFromTextFile<AdultData>("adult.data", hasHeader = false, separatorChar = ',')
        |> ml.Shuffle

    let testData =
        ml.Context.Data.LoadFromTextFile<AdultData>("adult.test", hasHeader = false, separatorChar = ',')
        |> ml.Shuffle

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
        |> Seq.map ml.Onehot // Create a one-hot encoder for each categorical column
        |> Seq.fold ml.Append (EstimatorChain()) // Add the encoders to a new EstimatorChain
        |> ml.Append <| ml.MapValue "Label" labelLookup "Label" // Map labels to either true or false
        |> ml.Append <| ml.Concatenate "Features" featureColumns // Concatenate feature columns into a single new column
        |> ml.Append <| ml.Normalize "Features" "FeaturesNorm" // Normalize features into a new column, FeaturesNorm

    let transformer =
        pipeline
        |> ml.Fit trainData // Fit our pipeline on the training data

    // Print transformed data
    do
        let transformedData =
            trainData
            |> ml.Transform transformer

        printfn "------------------\nData As Loaded\n------------------"
        ml.Context.Data.CreateEnumerable<AdultData>(trainData, reuseRowObject = false)
        |> Seq.take 3
        |> Seq.iter (printfn "%A")

        printfn "------------------\nTransformed Data\n------------------"
        ml.Context.Data.CreateEnumerable<AdultDataTransformed>(transformedData, reuseRowObject = false)
        |> Seq.take 3
        |> Seq.iter (printfn "%A")

    let estimator =
        ml.Context.BinaryClassification.Trainers.SdcaLogisticRegression(featureColumnName = "FeaturesNorm")
        |> ml.DowncastEstimator

    let model =
        trainData // Begin with the training data
        |> ml.Transform transformer // Transform using the transformer built above
        |> ml.CrossValidateBinaryClassification estimator 3 // 3-fold cross-validation
        |> ml.PrintBinaryClassificationCvMetrics // Print cross-fold metrics
        |> Seq.maxBy (fun cvResult -> cvResult.Metrics.Accuracy) // Select the best model by R-squared
        |> fun cvResult -> cvResult.Model

    do
        model
        |> ml.Transform <| ml.Transform transformer testData // Transform the test data and get predictions
        |> ml.Context.BinaryClassification.Evaluate // Get test metrics
        |> ml.PrintBinaryClassificationMetrics

    // Show some sample predictions
    let sampleData =
        testData
        |> ml.Transform transformer

    let predictionEngine = ml.Context.Model.CreatePredictionEngine<AdultDataTransformed, AdultPrediction>(model)

    do
        printfn "------------------\nSample Predictions\n------------------"

        ml.Context.Data.CreateEnumerable<AdultDataTransformed>(sampleData, reuseRowObject = false)
        |> Seq.take 5
        |> Seq.map predictionEngine.Predict
        |> Seq.iter (printfn "%A")

    0
