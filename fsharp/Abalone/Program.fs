open Microsoft.ML
open Microsoft.ML.Data
open Abalone
open System.IO
open System.Net
open FunctionalMl


[<EntryPoint>]
let main argv =
    if not <| File.Exists("ablone.data") then
        use client = new WebClient()
        client.DownloadFile("http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", "abalone.data")
    
    let ml = MlWrapper()
    
    let allDataView = ml.Context.Data.LoadFromTextFile<AbaloneData>("abalone.data", hasHeader = false, separatorChar = ',')
    
    let trainDataView, testDataView =
        ml.Shuffle allDataView
        |> ml.Split 0.2
    
    let featureColumns = [| "Sex"; "Length"; "Diameter"; "Height"; "WholeWeight"; "ShuckedWeight"; "VisceraWeight"; "ShellWeight" |]

    let pipeline = 
        EstimatorChain()
        |> ml.Append <| ml.Onehot "Sex" // one-hot encode the Sex feature
        |> ml.Append <| ml.Concatenate "Features" featureColumns // Concatenate feature columns into a single new column
        |> ml.Append <| ml.Normalize "Features" "FeaturesNorm" // Normalize features into a new column, FeaturesNorm

    let transformer =
        pipeline
        |> ml.Fit trainDataView // Fit our pipeline on the training data

    // Print transformed data
    do
        let transformedData =
            trainDataView
            |> ml.Transform transformer

        printfn "------------------\nData As Loaded\n------------------"
        ml.Context.Data.CreateEnumerable<AbaloneData>(trainDataView, reuseRowObject = false)
        |> Seq.take 3
        |> Seq.iter (printfn "%A")

        printfn "------------------\nTransformed Data\n------------------"
        ml.Context.Data.CreateEnumerable<AbaloneDataTransformed>(transformedData, reuseRowObject = false)
        |> Seq.take 3
        |> Seq.iter (printfn "%A")


    let estimator = 
        ml.Context.Regression.Trainers.LbfgsPoissonRegression(featureColumnName = "FeaturesNorm")
        |> ml.DowncastEstimator
    
    let model =
        trainDataView // Begin with the training data
        |> ml.Transform transformer // Transform using the transformer built above
        |> ml.CrossValidate estimator 3 // 3-fold cross-validation
        |> ml.PrintRegressionCvMetrics // Print cross-fold metrics
        |> Seq.maxBy (fun cvResult -> cvResult.Metrics.RSquared) // Select the best model by R-squared
        |> fun cvResult -> cvResult.Model

    do
        model
        |> ml.Transform <| ml.Transform transformer testDataView // Transform the test data and get predictions
        |> ml.Context.Regression.Evaluate // Get test metrics
        |> ml.PrintRegressionMetrics

    let sampleData =
        allDataView
        |> ml.Shuffle 
        |> ml.Transform transformer

    let predictions = model.Transform(sampleData)

    let predictionRecords = ml.Context.Data.CreateEnumerable<AbalonePrediction>(predictions, reuseRowObject = false)

    do
        printfn "------------------\nSample Predictions\n------------------"
        predictionRecords
        |> Seq.take 10
        |> Seq.iter (printfn "%A")

    0
