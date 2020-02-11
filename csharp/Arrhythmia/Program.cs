using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.IO;
using System.Linq;
using System.Net;

namespace Arrhythmia
{
    class Program
    {
        static void Main(string[] args)
        {
            if (!File.Exists("arrhythmia.data"))
            {
                using var client = new WebClient();
                client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/arrhythmia/arrhythmia.data", "arrhythmia.data");
            }

            var context = new MLContext();

            var columns = Enumerable.Range(0, 278)
                .Select(n => new TextLoader.Column($"{n}", DataKind.Single, n))
                .ToList();
            columns.Add(new TextLoader.Column("Label", DataKind.Int32, 279));

            var featureColumns = Enumerable.Range(0, 278).Select(n => $"{n}").ToArray();

            var textLoader = context.Data.CreateTextLoader(columns.ToArray(), hasHeader: false, separatorChar: ',');
            var allData = textLoader.Load("arrhythmia.data");
            allData = context.Data.ShuffleRows(allData);

            var splitData = context.Data.TrainTestSplit(allData, testFraction: 0.2);
            var (trainData, testData) = (splitData.TrainSet, splitData.TestSet);

            Console.WriteLine("---------------------\nTarget Value Counts\n--------------------");
            var labelGroups = context.Data.CreateEnumerable<ArrhythmiaLabel>(allData, reuseRowObject: false)
                .GroupBy(x => x.Label);
            foreach (var labelGroup in labelGroups)
            {
                Console.WriteLine($"{labelGroup.Key}: {labelGroup.Count()}");
            }

            // Try normalizing and using PCA with different trainers to see impacts
            var pipeline = context.Transforms.Conversion
                .MapValueToKey("Label", "Label")
                .Append(context.Transforms.Conversion.MapKeyToValue("LabelValue", "Label"))
                .Append(context.Transforms.Concatenate("Features", featureColumns))
                .Append(context.Transforms.ReplaceMissingValues("Features", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean));
                //.Append(context.Transforms.NormalizeLogMeanVariance("FeaturesNorm", "Features"))
                //.Append(context.Transforms.ProjectToPrincipalComponents("Features", "FeaturesNorm", rank: 10, ensureZeroMean: false));

            var transformer = pipeline.Fit(trainData);

            // Print transformed data
            Console.WriteLine("------------------\nData As Loaded\n------------------");
            var sourceRows = trainData.Preview().RowView.Take(3);
            foreach (var row in sourceRows)
            {
                Console.Write("{ ");
                foreach (var kv in row.Values)
                {
                    Console.Write($"{kv.Value}, ");
                }
                Console.WriteLine(" }");
            }

            Console.WriteLine("------------------\nTransformed Data\n------------------");
            var transformedData = transformer.Transform(trainData);
            var transformedItems = context.Data
                .CreateEnumerable<ArrhythmiaDataTransformed>(transformedData, reuseRowObject: false)
                .Take(3);
            foreach (var item in transformedItems)
            {
                Console.WriteLine(item);
            }

            var estimator = context.MulticlassClassification.Trainers.LightGbm(featureColumnName: "Features", learningRate: 0.1);

            var transformedTrainData = transformer.Transform(trainData);
            var cvResults = context.MulticlassClassification.CrossValidate(transformedTrainData, estimator, numberOfFolds: 5);
            var cvResult = cvResults
                .OrderByDescending(x => x.Metrics.MicroAccuracy)
                .First();

            Console.WriteLine("------------------\nCross Validation Metrics\n------------------");
            Console.WriteLine($"MacroAccuracy: {cvResults.Average(x => x.Metrics.MacroAccuracy)}");
            Console.WriteLine($"MicroAccuracy: {cvResults.Average(x => x.Metrics.MicroAccuracy)}");
            Console.WriteLine($"Log Loss: {cvResults.Average(x => x.Metrics.LogLoss)}");
            Console.WriteLine($"Confusion Matrix:");
            Console.WriteLine(cvResult.Metrics.ConfusionMatrix.GetFormattedConfusionTable());
            Console.WriteLine("--------------------------------");
            Console.WriteLine();

            var transformedTestData = transformer.Transform(testData);
            var predictions = cvResult.Model.Transform(transformedTestData);
            var metrics = context.MulticlassClassification.Evaluate(predictions);

            Console.WriteLine("------------------\n Test Metrics\n------------------");
            Console.WriteLine($"MacroAccuracy: {metrics.MacroAccuracy}");
            Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy}");
            Console.WriteLine($"Log Loss: {metrics.LogLoss}");
            Console.WriteLine($"Confusion Matrix:");
            Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());
            Console.WriteLine("--------------------------------");
            Console.WriteLine();

            // Show some sample predictions
            var sampleData = context.Data.ShuffleRows(testData);
            var transformedSampleData = transformer.Transform(sampleData);

            Console.WriteLine("------------------\nSample Predictions\n------------------");
            var samplePredictions = cvResult.Model.Transform(transformedSampleData);
            var mapValues = context.Transforms.Conversion
                .MapKeyToValue("PredictedLabelValue", "PredictedLabel")
                .Append(context.Transforms.Conversion.MapKeyToValue("LabelValue", "Label"))
                .Fit(samplePredictions);
            samplePredictions = mapValues.Transform(samplePredictions);
            var samplePredictionItems = context.Data.CreateEnumerable<ArrhythmiaPrediction>(samplePredictions, reuseRowObject: false);

            foreach (var item in samplePredictionItems.Take(5))
            {
                Console.WriteLine(item);
            }
        }
    }
}
