namespace FunctionalMl
open Microsoft.ML
open Microsoft.ML.Data
open System.Collections.Generic

type MlWrapper() =
    let context = MLContext()

    member _.Context
        with get() = context

    member _.Append (chain : EstimatorChain<'T>) transform =
        chain.Append(transform)

    member _.Shuffle dataView =
        context.Data.ShuffleRows(dataView)

    member _.Split testFraction dataView =
        let splitData = context.Data.TrainTestSplit(dataView, testFraction = testFraction)
        splitData.TrainSet, splitData.TestSet

    member _.Onehot (column : string) =
        context.Transforms.Categorical.OneHotEncoding(column)

    member _.Concatenate outputColumnName inputColumnNames =
        context.Transforms.Concatenate(outputColumnName = outputColumnName, inputColumnNames = inputColumnNames)

    member _.MapValue outputColumnName (keyValuePairs : IEnumerable<KeyValuePair<string, bool>>) inputColumnName =
        context.Transforms.Conversion.MapValue(outputColumnName, keyValuePairs, inputColumnName)

    member _.NormalizeLp inputColumn outputColumn  =
        context.Transforms.NormalizeLpNorm(outputColumnName = outputColumn, inputColumnName = inputColumn)

    member _.NormalizeMinMax inputColumn outputColumn  =
        context.Transforms.NormalizeMinMax(outputColumnName = outputColumn, inputColumnName = inputColumn)

    member _.Fit<'T when 'T :> ITransformer and 'T : not struct > dataView (pipeline : EstimatorChain<'T>) =
        pipeline.Fit(dataView)

    member _.DowncastEstimator (e : IEstimator<'a>) =
        match e with
        | :? IEstimator<ITransformer> as p -> p
        | _ -> failwith "The estimator has to be an instance of IEstimator<ITransformer>."

    member _.Transform (transformer : ITransformer) dataView =
        transformer.Transform(dataView)

    member _.CrossValidateRegression estimator numberOfFolds dataView =
        context.Regression.CrossValidate(dataView, estimator, numberOfFolds = numberOfFolds)

    member _.CrossValidateBinaryClassification estimator numberOfFolds dataView =
        context.BinaryClassification.CrossValidate(dataView, estimator, numberOfFolds = numberOfFolds)

    member _.PrintRegressionMetrics (metrics : RegressionMetrics) =
        printfn "------------------\nTest Metrics\n------------------"
        printfn "Mean Absolute Error: %f" metrics.MeanAbsoluteError
        printfn "Mean Squared Error: %f" metrics.MeanSquaredError
        printfn "Root Mean Squared Error: %f" metrics.RootMeanSquaredError
        printfn "R-squared: %f" metrics.RSquared

    member _.PrintBinaryClassificationMetrics (metrics : CalibratedBinaryClassificationMetrics) =
        printfn "------------------\nTest Metrics\n------------------"
        printfn "Accuracy: %f" metrics.Accuracy
        printfn "Log Loss: %f" metrics.LogLoss
        printfn "Area Under ROC Curve: %f" metrics.AreaUnderRocCurve
        printfn "F1 Score: %f" metrics.F1Score

    member _.PrintRegressionCvMetrics (cvResults : TrainCatalogBase.CrossValidationResult<RegressionMetrics> seq) =
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

    member _.PrintBinaryClassificationCvMetrics(cvResults : TrainCatalogBase.CrossValidationResult<CalibratedBinaryClassificationMetrics> seq) =
        printfn "------------------\nCross Validation Metrics\n------------------"
        cvResults
        |> Seq.map (fun cvResult -> cvResult.Metrics.Accuracy)
        |> Seq.average
        |> printfn "Accuracy: %f"; cvResults
        |> Seq.map (fun cvResult -> cvResult.Metrics.AreaUnderRocCurve)
        |> Seq.average
        |> printfn "Area Under Roc Curve: %f"; cvResults
        |> Seq.map (fun cvResult -> cvResult.Metrics.F1Score)
        |> Seq.average
        |> printfn "F1 Score: %f"

        cvResults
