open Microsoft.ML
open Microsoft.ML.Data
open Abalone
open System.IO
open System.Net


let printCvResultMetrics (cvResults : TrainCatalogBase.CrossValidationResult<RegressionMetrics> seq) =
    do
        printfn "------------------\nCross Validation Metrics\n------------------"
        cvResults
        |> Seq.map (fun cvResult -> cvResult.Metrics.MeanAbsoluteError)
        |> Seq.average
        |> printfn "Mean Absolute Error: %f"; cvResults
        |> Seq.map (fun cvResult -> cvResult.Metrics.MeanSquaredError)
        |> Seq.average
        |> printfn "Mean Squared Error: %f"; cvResults
        |> Seq.map (fun cvResult -> cvResult.Metrics.RootMeanSquaredError)
        |> Seq.average
        |> printfn "Root Mean Squared Error: %f"; cvResults
        |> Seq.map (fun cvResult -> cvResult.Metrics.RSquared)
        |> Seq.average
        |> printfn "R-squared: %f"

    cvResults

let printMetrics (metrics : RegressionMetrics) =
    do
        printfn "------------------\nTest Metrics\n------------------"
        printfn "Mean Absolute Error: %f" metrics.MeanAbsoluteError
        printfn "Mean Squared Error: %f" metrics.MeanSquaredError
        printfn "Root Mean Squared Error: %f" metrics.RootMeanSquaredError
        printfn "R-squared: %f" metrics.RSquared

let shuffle (context : MLContext) dataView =
    context.Data.ShuffleRows(dataView)

let split (context : MLContext) testFraction dataView =
    let splitData = context.Data.TrainTestSplit(dataView, testFraction = testFraction)
    splitData.TrainSet, splitData.TestSet

let append (chain : EstimatorChain<'T>) transform =
    chain.Append(transform)
    
let onehot (context : MLContext) (column : string) =
    context.Transforms.Categorical.OneHotEncoding(column)
    
let concatenate (context : MLContext) outputColumnName inputColumnNames =
    context.Transforms.Concatenate(outputColumnName = outputColumnName, inputColumnNames = inputColumnNames)
    
let normalize (context : MLContext) inputColumn outputColumn =
    context.Transforms.NormalizeLpNorm(outputColumnName = outputColumn, inputColumnName = inputColumn)
    
let downcastEstimator (e : IEstimator<'a>) =
    match e with
    | :? IEstimator<ITransformer> as p -> p
    | _ -> failwith "The estimator has to be an instance of IEstimator<ITransformer>."

let makeEstimator (context : MLContext) featureColumnName =
    context.Regression.Trainers.LbfgsPoissonRegression(featureColumnName = featureColumnName)
    |> downcastEstimator

let transform (transformer : ITransformer) dataView =
    transformer.Transform(dataView)

let crossValidate (context : MLContext) estimator numberOfFolds dataView =
    context.Regression.CrossValidate(dataView, estimator, numberOfFolds = numberOfFolds)


[<EntryPoint>]
let main argv =
    if not <| File.Exists("ablone.data") then
        use client = new WebClient()
        client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", "abalone.data")
    
    let context = new MLContext()
    
    let allDataView = context.Data.LoadFromTextFile<AbaloneData>("abalone.data", hasHeader = false, separatorChar = ',')
    
    let trainDataView, testDataView =
        shuffle context allDataView
        |> split context 0.2
    
    let featureColumns = [| "Sex"; "Length"; "Diameter"; "Height"; "WholeWeight"; "ShuckedWeight"; "VisceraWeight"; "ShellWeight" |]

    let transformer =
        EstimatorChain()
        |> append <| onehot context "Sex" // one-hot encode the Sex feature
        |> append <| concatenate context "Features" featureColumns // Concatenate feature columns into a single new column
        |> append <| normalize context "Features" "FeaturesNorm" // Normalize features into a new column, FeaturesNorm
        |> (fun pipeline -> pipeline.Fit(trainDataView)) // Fit our pipeline on the training data

    let estimator = makeEstimator context "FeaturesNorm"
    
    do
        trainDataView // Begin with the training data
        |> transform transformer // Transform using the transformer built above
        |> crossValidate context estimator 3 // 3-fold cross-validation
        |> printCvResultMetrics // Print cross-fold metrics
        |> Seq.maxBy (fun cvResult -> cvResult.Metrics.RSquared) // Select the best model by R-squared
        |> fun cvResult -> cvResult.Model
        |> transform <| transform transformer testDataView // Transform the test data and get predictions
        |> context.Regression.Evaluate // Get test metrics
        |> printMetrics

    0
