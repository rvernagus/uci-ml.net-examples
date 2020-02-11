using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.IO;
using System.Linq;
using System.Net;

namespace Annealing
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            if (!File.Exists("anneal.data"))
            {
                using var client = new WebClient();
                client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/annealing/anneal.data", "anneal.data");
            }

            if (!File.Exists("anneal.test"))
            {
                using var client = new WebClient();
                client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/annealing/anneal.test", "anneal.test");
            }

            var context = new MLContext();

            var trainData = context.Data.LoadFromTextFile<AnnealData>("anneal.data", hasHeader: false, separatorChar: ',');
            trainData = context.Data.ShuffleRows(trainData);

            var testData = context.Data.LoadFromTextFile<AnnealData>("anneal.test", hasHeader: false, separatorChar: ',');
            testData = context.Data.ShuffleRows(testData);

            var featureColumns = new[]
            {
                nameof(AnnealData.Family), nameof(AnnealData.ProductType), nameof(AnnealData.Steel), nameof(AnnealData.Carbon), nameof(AnnealData.Hardness),
                nameof(AnnealData.TemperRolling), nameof(AnnealData.Condition), nameof(AnnealData.Formability), nameof(AnnealData.Strength), nameof(AnnealData.NonAgeing),
                nameof(AnnealData.SurfaceFinish), nameof(AnnealData.SurfaceQuality), nameof(AnnealData.Enamelability), nameof(AnnealData.Bc), nameof(AnnealData.Bf),
                nameof(AnnealData.Bt), nameof(AnnealData.BwMe), nameof(AnnealData.Bl), nameof(AnnealData.M), nameof(AnnealData.Chrom), nameof(AnnealData.Phos),
                nameof(AnnealData.Cbond), nameof(AnnealData.Marvi), nameof(AnnealData.Exptl), nameof(AnnealData.Ferro), nameof(AnnealData.Corr),
                nameof(AnnealData.BlueBrightVarnClean), nameof(AnnealData.Lustre), nameof(AnnealData.Jurofm), nameof(AnnealData.S), nameof(AnnealData.P), nameof(AnnealData.Shape),
                nameof(AnnealData.Thick), nameof(AnnealData.Width), nameof(AnnealData.Len), nameof(AnnealData.Oil), nameof(AnnealData.Bore), nameof(AnnealData.Packing)
            };

            var categoricalColumns = new[]
            {
                nameof(AnnealData.Family), nameof(AnnealData.ProductType), nameof(AnnealData.Steel), nameof(AnnealData.TemperRolling), nameof(AnnealData.Condition),
                nameof(AnnealData.Formability), nameof(AnnealData.NonAgeing), nameof(AnnealData.SurfaceFinish), nameof(AnnealData.SurfaceQuality), nameof(AnnealData.Enamelability),
                nameof(AnnealData.Bc), nameof(AnnealData.Bf), nameof(AnnealData.Bt), nameof(AnnealData.BwMe), nameof(AnnealData.Bl), nameof(AnnealData.M), nameof(AnnealData.Chrom),
                nameof(AnnealData.Phos), nameof(AnnealData.Cbond), nameof(AnnealData.Marvi), nameof(AnnealData.Exptl), nameof(AnnealData.Ferro), nameof(AnnealData.Corr),
                nameof(AnnealData.BlueBrightVarnClean), nameof(AnnealData.Lustre), nameof(AnnealData.Jurofm), nameof(AnnealData.S), nameof(AnnealData.P), nameof(AnnealData.Shape),
                nameof(AnnealData.Oil), nameof(AnnealData.Bore), nameof(AnnealData.Packing)
            };

            var chain = new EstimatorChain<OneHotEncodingTransformer>();
            var pipeline = categoricalColumns
                .Aggregate(chain, (pl, col) => pl.Append(context.Transforms.Categorical.OneHotEncoding(col)))
                .Append(context.Transforms.Conversion.MapValueToKey("Label", "Label"))
                .Append(context.Transforms.Conversion.MapKeyToValue("LabelValue", "Label"))
                .Append(context.Transforms.Concatenate("Features", featureColumns));

            var transformer = pipeline.Fit(trainData);

            // Print transformed data
            Console.WriteLine("------------------\nData As Loaded\n------------------");
            var sourceItems = context.Data
                .CreateEnumerable<AnnealData>(trainData, reuseRowObject: false)
                .Take(3);
            foreach (var item in sourceItems)
            {
                Console.WriteLine(item);
            }

            Console.WriteLine("------------------\nTransformed Data\n------------------");
            var transformedData = transformer.Transform(trainData);
            var transformedItems = context.Data
                .CreateEnumerable<AnnealDataTransformed>(transformedData, reuseRowObject: false)
                .Take(3);
            foreach (var item in transformedItems)
            {
                Console.WriteLine(item);
            }

            var estimator = context.MulticlassClassification.Trainers.LbfgsMaximumEntropy(featureColumnName: "Features");

            var transformedTrainData = transformer.Transform(trainData);
            var cvResults = context.MulticlassClassification.CrossValidate(transformedTrainData, estimator, numberOfFolds: 3);
            var cvResult = cvResults
                .OrderByDescending(x => x.Metrics.MacroAccuracy)
                .First();

            Console.WriteLine("------------------\nCross Validation Metrics\n------------------");
            Console.WriteLine($"Accuracy: {cvResults.Average(x => x.Metrics.MacroAccuracy)}");
            Console.WriteLine($"Log Loss: {cvResults.Average(x => x.Metrics.LogLoss)}");
            Console.WriteLine($"Confusion Matrix:");
            Console.WriteLine(cvResult.Metrics.ConfusionMatrix.GetFormattedConfusionTable());
            Console.WriteLine("--------------------------------");
            Console.WriteLine();

            var transformedTestData = transformer.Transform(testData);
            var predictions = cvResult.Model.Transform(transformedTestData);
            var metrics = context.MulticlassClassification.Evaluate(predictions);

            Console.WriteLine("------------------\n Test Metrics\n------------------");
            Console.WriteLine($"Accuracy: {metrics.MacroAccuracy}");
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
            var samplePredictionItems = context.Data.CreateEnumerable<AnnealPrediction>(samplePredictions, reuseRowObject: false);

            foreach (var item in samplePredictionItems.Take(5))
            {
                Console.WriteLine(item);
            }
        }
    }
}
