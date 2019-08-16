open System.Net
open System.IO
open Microsoft.ML
open Autos
open Microsoft.ML.Data
open Microsoft.ML.Transforms


let onehot (context : MLContext) (column : string) =
    context.Transforms.Categorical.OneHotEncoding(column)

let append (chain : EstimatorChain<'T>) transform =
    chain.Append(transform)

let concatenate (context : MLContext) outputColumnName inputColumnNames =
    context.Transforms.Concatenate(outputColumnName = outputColumnName, inputColumnNames = inputColumnNames)

let normalize (context : MLContext) inputColumn outputColumn =
    context.Transforms.NormalizeBinning(outputColumnName = outputColumn, inputColumnName = inputColumn)

let replaceMissing (context : MLContext) (inputColumn : string) (outputColumn : string) =
    context.Transforms.ReplaceMissingValues(inputColumn, replacementMode = MissingValueReplacingEstimator.ReplacementMode.Mean)

let downcastEstimator (e : IEstimator<'a>) =
    match e with
    | :? IEstimator<ITransformer> as p -> p
    | _ -> failwith "The estimator has to be an instance of IEstimator<ITransformer>."

let printMetrics (metrics : RegressionMetrics) =
    printfn "Mean Absolute Error: %f" metrics.MeanAbsoluteError
    printfn "Mean Squared Error: %f" metrics.MeanSquaredError
    printfn "Root Mean Squared Error: %f" metrics.RootMeanSquaredError
    printfn "R Squared: %f" metrics.RSquared


[<EntryPoint>]
let main argv =
    if not <| File.Exists("imports-85.data") then
        use client = new WebClient()
        client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data", "imports-85.data")
    
    let context = new MLContext()

    let allDataView =
        context.Data.LoadFromTextFile<AutosData>("imports-85.data", hasHeader = false, separatorChar = ',')
        |> context.Data.ShuffleRows

    let trainDataView, testDataView =
        let split = context.Data.TrainTestSplit(allDataView, testFraction = 0.2)
        split.TrainSet, split.TestSet

    let featureColumns =
        [|
            "Symboling"; "NormLosses"; "Make"; "FuelType"; "Aspiration"; "NumDoors"; "BodyStyle"; "DriveWheels"; "EngineLoc";
            "WheelBase"; "Length"; "Width"; "Height"; "CurbWeight"; "EngineType"; "NumCylinders"; "EngineSize"; "FuelSystem";
            "Bore"; "Stroke"; "CompressionRatio"; "Horsepower"; "PeakRpm"; "CityMpg"; "HighwayMpg"
        |]

    let categoricalColumns =
        [|
            "Symboling"; "Make"; "FuelType"; "Aspiration"; "NumDoors"; "BodyStyle"; "DriveWheels"; "EngineLoc"; "EngineType";
            "NumCylinders"; "FuelSystem"
        |]

    let transformer =
        categoricalColumns
        |> Seq.map (onehot context)
        |> Seq.fold append (EstimatorChain())
        |> append <| concatenate context "Features" featureColumns
        |> append <| replaceMissing context "Features" "Features"
        |> append <| normalize context "Features" "FeaturesNorm"
        |> fun pipeline -> pipeline.Fit(trainDataView)

    let transformedTrainData = transformer.Transform(trainDataView)
    let transformedTestData = transformer.Transform(testDataView)

    let estimator = context.Regression.Trainers.Sdca(featureColumnName = "FeaturesNorm")
    let finalEstimator = downcastEstimator estimator

    let cvResult =
        context.Regression.CrossValidate(transformedTrainData, finalEstimator, numberOfFolds = 3)
        |> Seq.maxBy (fun cvResult -> cvResult.Metrics.RSquared)

    do
        cvResult.Model.Transform(transformedTestData)
        |> context.Regression.Evaluate
        |> printMetrics

    let testRecords =
        allDataView
        |> context.Data.ShuffleRows
        |> fun data -> context.Data.TakeRows(data, 5L)
        |> fun rows -> context.Data.CreateEnumerable<AutosData>(rows, reuseRowObject = true)

    let actualPrices =
        testRecords
        |> Seq.map (fun record -> record.Price)

    let predictedPrices =
        testRecords
        |> context.Data.LoadFromEnumerable
        |> transformer.Transform
        |> cvResult.Model.Transform
        |> fun resultView -> resultView.Preview().RowView
        |> Seq.map (fun record -> record.Values.[51])

    do
        printfn "------------------"
        Seq.zip actualPrices predictedPrices
        |> Seq.iter (printfn "%A")

    0
