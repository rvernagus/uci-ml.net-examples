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

            var transformer = context
                .Transforms.Categorical.OneHotEncoding(nameof(AbaloneData.Sex))
                .Append(context.Transforms.Concatenate("Features", featureColumns))
                .Append(context.Transforms.NormalizeLpNorm("FeaturesNorm", "Features"))
                .Fit(allData);

            var estimator = context.Regression.Trainers.LbfgsPoissonRegression(featureColumnName: "FeaturesNorm");

            var transformedTrainData = transformer.Transform(trainData);
            var cvResults = context.Regression.CrossValidate(transformedTrainData, estimator, numberOfFolds: 5);
            var cvResult = cvResults
                .OrderByDescending(x => x.Metrics.RSquared)
                .First();

            Console.WriteLine($"Mean Absolute Error: {cvResults.Average(x => x.Metrics.MeanAbsoluteError)}");
            Console.WriteLine($"Mean Squared Error: {cvResults.Average(x => x.Metrics.MeanSquaredError)}");
            Console.WriteLine($"Root Mean Squared Error: {cvResults.Average(x => x.Metrics.RootMeanSquaredError)}");
            Console.WriteLine($"R-squared: {cvResults.Average(x => x.Metrics.RSquared)}");
            Console.WriteLine("--------------------------------");
            Console.WriteLine();

            var transformedTestData = transformer.Transform(testData);
            var predictions = cvResult.Model.Transform(transformedTestData);
            var metrics = context.Regression.Evaluate(predictions);

            Console.WriteLine($"Mean Absolute Error: {metrics.MeanAbsoluteError}");
            Console.WriteLine($"Mean Squared Error: {metrics.MeanSquaredError}");
            Console.WriteLine($"Root Mean Squared Error: {metrics.RootMeanSquaredError}");
            Console.WriteLine($"R-squared: {metrics.RSquared}");
            Console.WriteLine("--------------------------------");
            Console.WriteLine();

            context.Data
                .CreateEnumerable<AbalonePrediction>(predictions, reuseRowObject: false)
                .Take(10)
                .ToList()
                .ForEach(Console.WriteLine);
        }
    }
}
