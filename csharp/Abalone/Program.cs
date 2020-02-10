using Microsoft.ML;
using System;
using System.IO;
using System.Linq;
using System.Net;

namespace Abalone
{
    class Program
    {
        static void Main(string[] args)
        {
            if (!File.Exists("abalone.data"))
            {
                using var client = new WebClient();
                client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data", "abalone.data");
            }

            var context = new MLContext();

            var allData = context.Data.LoadFromTextFile<AbaloneData>("abalone.data", hasHeader: false, separatorChar: ',');
            allData = context.Data.ShuffleRows(allData);

            var splitData = context.Data.TrainTestSplit(allData, testFraction: 0.2);
            var (trainData, testData) = (splitData.TrainSet, splitData.TestSet);

            var featureColumns = new[]
            {
                nameof(AbaloneData.Sex), nameof(AbaloneData.Length), nameof(AbaloneData.Diameter), nameof(AbaloneData.Height),
                nameof(AbaloneData.WholeWeight), nameof(AbaloneData.ShuckedWeight), nameof(AbaloneData.VisceraWeight),
                nameof(AbaloneData.ShellWeight)
            };

            var pipeline = context
                .Transforms.Categorical.OneHotEncoding(nameof(AbaloneData.Sex))
                .Append(context.Transforms.Concatenate("Features", featureColumns))
                .Append(context.Transforms.NormalizeLpNorm("FeaturesNorm", "Features"));

            var transformer = pipeline.Fit(trainData);

            // Print transformed data
            Console.WriteLine("------------------\nData As Loaded\n------------------");
            var sourceItems = context.Data
                .CreateEnumerable<AbaloneData>(trainData, reuseRowObject: false)
                .Take(3);
            foreach (var item in sourceItems)
            {
                Console.WriteLine(item);
            }

            Console.WriteLine("------------------\nTransformed Data\n------------------");
            var transformedData = transformer.Transform(trainData);
            var transformedItems = context.Data
                .CreateEnumerable<AbaloneDataTransformed>(transformedData, reuseRowObject: false)
                .Take(3);
            foreach (var item in transformedItems)
            {
                Console.WriteLine(item);
            }

            var estimator = context.Regression.Trainers.LbfgsPoissonRegression(featureColumnName: "FeaturesNorm");

            var transformedTrainData = transformer.Transform(trainData);
            var cvResults = context.Regression.CrossValidate(transformedTrainData, estimator, numberOfFolds: 3);
            var cvResult = cvResults
                .OrderByDescending(x => x.Metrics.RSquared)
                .First();

            Console.WriteLine("------------------\nCross Validation Metrics\n------------------");
            Console.WriteLine($"Mean Absolute Error: {cvResults.Average(x => x.Metrics.MeanAbsoluteError)}");
            Console.WriteLine($"Mean Squared Error: {cvResults.Average(x => x.Metrics.MeanSquaredError)}");
            Console.WriteLine($"Root Mean Squared Error: {cvResults.Average(x => x.Metrics.RootMeanSquaredError)}");
            Console.WriteLine($"R-squared: {cvResults.Average(x => x.Metrics.RSquared)}");
            Console.WriteLine("--------------------------------");
            Console.WriteLine();

            var transformedTestData = transformer.Transform(testData);
            var predictions = cvResult.Model.Transform(transformedTestData);
            var metrics = context.Regression.Evaluate(predictions);

            Console.WriteLine("------------------\nTest Metrics\n------------------");
            Console.WriteLine($"Mean Absolute Error: {metrics.MeanAbsoluteError}");
            Console.WriteLine($"Mean Squared Error: {metrics.MeanSquaredError}");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");
            Console.WriteLine($"R-squared: {metrics.RSquared}");
            Console.WriteLine("--------------------------------");
            Console.WriteLine();

            // Show some sample predictions
            var sampleData = context.Data.ShuffleRows(testData);
            var transformedSampleData = transformer.Transform(sampleData);

            var predictionEngine = context.Model.CreatePredictionEngine<AbaloneDataTransformed, AbalonePrediction>(cvResult.Model);

            Console.WriteLine("------------------\nSample Predictions\n------------------");
            var samplePredictions = context.Data.CreateEnumerable<AbaloneDataTransformed>(transformedSampleData, reuseRowObject: false)
                .Take(5)
                .Select(predictionEngine.Predict);
            foreach (var item in samplePredictions)
            {
                Console.WriteLine(item);
            }
        }
    }
}
