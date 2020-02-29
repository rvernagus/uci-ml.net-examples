namespace FunctionalMl
open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Transforms
open System.Collections.Generic

module ML =
    let context = MLContext()

    let append (chain : EstimatorChain<'T>) transform =
        chain.Append(transform)

    let concatenate outputColumn inputColumns =
        context.Transforms.Concatenate(outputColumnName = outputColumn, inputColumnNames = inputColumns)

    let crossValidateRegression estimator numberOfFolds dataView =
        context.Regression.CrossValidate(dataView, estimator, numberOfFolds = numberOfFolds)

    let crossValidateBinaryClassification estimator numberOfFolds dataView =
        context.BinaryClassification.CrossValidate(dataView, estimator, numberOfFolds = numberOfFolds)

    let crossValidateMulticlassClassification estimator numberOfFolds dataView =
        context.MulticlassClassification.CrossValidate(dataView, estimator, numberOfFolds = numberOfFolds)

    let downcastEstimator (e : IEstimator<'a>) =
        match e with
        | :? IEstimator<ITransformer> as p -> p
        | _ -> failwith "The estimator has to be an instance of IEstimator<ITransformer>."

    let fit<'T when 'T :> ITransformer and 'T : not struct > dataView (pipeline : EstimatorChain<'T>) =
        pipeline.Fit(dataView)

    let mapValue outputColumn (keyValuePairs : IEnumerable<KeyValuePair<string, bool>>) inputColumn =
        context.Transforms.Conversion.MapValue(outputColumn, keyValuePairs, inputColumn)

    let mapValueToKey inputColumn outputColumn =
        context.Transforms.Conversion.MapValueToKey(outputColumnName = outputColumn, inputColumnName = inputColumn)

    let mapKeyToValue inputColumn outputColumn =
        context.Transforms.Conversion.MapKeyToValue(outputColumn, inputColumnName = inputColumn)

    let normalizeLp inputColumn outputColumn  =
        context.Transforms.NormalizeLpNorm(outputColumnName = outputColumn, inputColumnName = inputColumn)

    let normalizeMinMax inputColumn outputColumn  =
        context.Transforms.NormalizeMinMax(outputColumnName = outputColumn, inputColumnName = inputColumn)

    let onehot (column : string) =
        context.Transforms.Categorical.OneHotEncoding(column)

    let pca inputColumnName outputColumnName rank =
        context.Transforms.ProjectToPrincipalComponents(outputColumnName = outputColumnName, inputColumnName = inputColumnName, rank = rank, ensureZeroMean = true)

    let replaceMissingWithDefault (inputColumn : string) (outputColumn : string) =
        context.Transforms.ReplaceMissingValues(inputColumn, replacementMode = MissingValueReplacingEstimator.ReplacementMode.DefaultValue)

    let replaceMissingWithMean (inputColumn : string) (outputColumn : string) =
        context.Transforms.ReplaceMissingValues(inputColumn, replacementMode = MissingValueReplacingEstimator.ReplacementMode.Mean)

    let shuffle dataView =
        context.Data.ShuffleRows(dataView)

    let split testFraction dataView =
        let splitData = context.Data.TrainTestSplit(dataView, testFraction = testFraction)
        splitData.TrainSet, splitData.TestSet

    let transform (transformer : ITransformer) dataView =
        transformer.Transform(dataView)

    let printRegressionMetrics (metrics : RegressionMetrics) =
        printfn "------------------\nTest Metrics\n------------------"
        printfn "Mean Absolute Error: %f" metrics.MeanAbsoluteError
        printfn "Mean Squared Error: %f" metrics.MeanSquaredError
        printfn "Root Mean Squared Error: %f" metrics.RootMeanSquaredError
        printfn "R-squared: %f" metrics.RSquared

    let printBinaryClassificationMetrics (metrics : CalibratedBinaryClassificationMetrics) =
        printfn "------------------\nTest Metrics\n------------------"
        printfn "Accuracy: %f" metrics.Accuracy
        printfn "Log Loss: %f" metrics.LogLoss
        printfn "Area Under ROC Curve: %f" metrics.AreaUnderRocCurve
        printfn "F1 Score: %f" metrics.F1Score

    let printMulticlassClassificationMetrics (metrics : MulticlassClassificationMetrics) =
        printfn "------------------\nTest Metrics\n------------------"
        printfn "Accuracy: %f" metrics.MacroAccuracy
        printfn "Log Loss: %f" metrics.LogLoss
        printfn "Confusion Matrix:"
        printfn "%s" <| metrics.ConfusionMatrix.GetFormattedConfusionTable()

    let printRegressionCvMetrics (cvResults : TrainCatalogBase.CrossValidationResult<RegressionMetrics> seq) =
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

    let printBinaryClassificationCvMetrics(cvResults : TrainCatalogBase.CrossValidationResult<CalibratedBinaryClassificationMetrics> seq) =
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

    let printMulticlassClassificationCvMetrics(cvResults: TrainCatalogBase.CrossValidationResult<MulticlassClassificationMetrics> seq) =
        printfn "------------------\nCross Validation Metrics\n------------------"
        cvResults
        |> Seq.map (fun cvResult -> cvResult.Metrics.MacroAccuracy)
        |> Seq.average
        |> printfn "Accuracy: %f"; cvResults
        |> Seq.map (fun cvResult -> cvResult.Metrics.LogLoss)
        |> Seq.average
        |> printfn "Log Loss: %f"

        cvResults
