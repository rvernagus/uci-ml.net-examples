open Microsoft.ML
open Microsoft.ML.Data
open Abalone
open System.IO
open System.Net


let printCvResultMetrics (cvResults : TrainCatalogBase.CrossValidationResult<RegressionMetrics> seq) =
    printfn "------------------\nCross Validation Metrics\n------------------"
    cvResults
    |> Seq.map (fun cvResult -> cvResult.Metrics.MeanAbsoluteError)
    |> Seq.average
    |> printfn "Mean Absolute Error: %f"
    cvResults
    |> Seq.map (fun cvResult -> cvResult.Metrics.MeanSquaredError)
    |> Seq.average
    |> printfn "Mean Squared Error: %f"
    cvResults
    |> Seq.map (fun cvResult -> cvResult.Metrics.RootMeanSquaredError)
    |> Seq.average
    |> printfn "Root Mean Squared Error: %f"
    cvResults
    |> Seq.map (fun cvResult -> cvResult.Metrics.RSquared)
    |> Seq.average
    |> printfn "R-squared: %f"
    cvResults

let printMetrics (metrics : RegressionMetrics) =
    printfn "------------------\nTest Metrics\n------------------"
    printfn "Mean Absolute Error: %f" metrics.MeanAbsoluteError
    printfn "Mean Squared Error: %f" metrics.MeanSquaredError
    printfn "Root Mean Squared Error: %f" metrics.RootMeanSquaredError
    printfn "R-squared: %f" metrics.RSquared

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


[<EntryPoint>]
let main argv =
    if not <| File.Exists("ablone.data") then
        use client = new WebClient()
        client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", "abalone.data")
    
    let context = new MLContext()
    
    let allDataView = context.Data.LoadFromTextFile<AbaloneData>("abalone.data", hasHeader = false, separatorChar = ',')
    
    let trainDataView, testDataView =
        let shuffledData = context.Data.ShuffleRows(allDataView)
        let split = context.Data.TrainTestSplit(shuffledData, testFraction = 0.2)
        split.TrainSet, split.TestSet
    
    let featureColumns = [| "Sex"; "Length"; "Diameter"; "Height"; "WholeWeight"; "ShuckedWeight"; "VisceraWeight"; "ShellWeight" |]

    let transformer =
        [ "Sex" ]
        |> Seq.map (onehot context)
        |> Seq.fold append (EstimatorChain())
        |> append <| concatenate context "Features" featureColumns
        |> append <| normalize context "Features" "FeaturesNorm"
        |> (fun pipeline -> pipeline.Fit(trainDataView))

    let transformedTrainData = transformer.Transform(trainDataView)
    let transformedTestData = transformer.Transform(testDataView)
    
    let estimator = context.Regression.Trainers.LbfgsPoissonRegression(featureColumnName = "FeaturesNorm")
    let finalEstimator = downcastEstimator estimator
    
    context.Regression.CrossValidate(transformedTrainData, finalEstimator, numberOfFolds = 3)
    |> printCvResultMetrics
    |> Seq.maxBy (fun cvResult -> cvResult.Metrics.RSquared)
    |> fun cvResult -> cvResult.Model
    |> fun model -> model.Transform(transformedTestData)
    |> context.Regression.Evaluate
    |> printMetrics

    0
