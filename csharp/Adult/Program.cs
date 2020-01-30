﻿using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;

namespace Adult
{
    class Program
    {
        static void Main(string[] args)
        {
            if (!File.Exists("adult.data"))
            {
                using var client = new WebClient();
                client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", "adult.data");
            }

            if (!File.Exists("abalone.test"))
            {
                using var client = new WebClient();
                client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", "adult.test");
            }

            var context = new MLContext();

            var trainData = context.Data.LoadFromTextFile<AdultData>("adult.data", hasHeader: false, separatorChar: ',');
            trainData = context.Data.ShuffleRows(trainData);

            var testData = context.Data.LoadFromTextFile<AdultData>("adult.test", hasHeader: false, separatorChar: ',');
            testData = context.Data.ShuffleRows(testData);

            var featureColumns = new[]
            {
                nameof(AdultData.Age), nameof(AdultData.WorkClass), nameof(AdultData.Fnlwgt), nameof(AdultData.Education),
                nameof(AdultData.EducationNum), nameof(AdultData.MaritalStatus), nameof(AdultData.Occupation), nameof(AdultData.Relationship),
                nameof(AdultData.Race), nameof(AdultData.Sex), nameof(AdultData.CapitalGain), nameof(AdultData.CapitalLoss),
                nameof(AdultData.HoursPerWeek), nameof(AdultData.NativeCountry)
            };

            var categoricalColumns = new[]
            {
                nameof(AdultData.WorkClass), nameof(AdultData.Education), nameof(AdultData.MaritalStatus), nameof(AdultData.Occupation),
                nameof(AdultData.Relationship), nameof(AdultData.Race), nameof(AdultData.Sex), nameof(AdultData.NativeCountry)
            };

            var labelLookup = new Dictionary<string, bool>
            {
                ["<=50K"] = false,
                ["<=50K."] = false,
                [">50K"] = true,
                [">50K."] = true
            };

            var chain = new EstimatorChain<Microsoft.ML.Transforms.OneHotEncodingTransformer>();
            var transformer = categoricalColumns
                .Aggregate(chain, (pl, col) => pl.Append(context.Transforms.Categorical.OneHotEncoding(col)))
                .Append(context.Transforms.Conversion.MapValue("Label", labelLookup, "Label"))
                .Append(context.Transforms.Concatenate("Features", featureColumns))
                .Append(context.Transforms.NormalizeBinning("FeaturesNorm", "Features"))
                .Fit(trainData);

            var estimator = context.BinaryClassification.Trainers.SdcaLogisticRegression(featureColumnName: "FeaturesNorm");

            var transformedTrainData = transformer.Transform(trainData);
            var cvResults = context.BinaryClassification.CrossValidate(transformedTrainData, estimator, numberOfFolds: 3);
            var cvResult = cvResults
                .OrderByDescending(x => x.Metrics.Accuracy)
                .First();

            Console.WriteLine($"Accuracy: {cvResults.Average(x => x.Metrics.Accuracy)}");
            Console.WriteLine($"Area Under Roc Curve: {cvResults.Average(x => x.Metrics.AreaUnderRocCurve)}");
            Console.WriteLine($"F1 Score: {cvResults.Average(x => x.Metrics.F1Score)}");
            Console.WriteLine("--------------------------------");
            Console.WriteLine();

            var transformedTestData = transformer.Transform(testData);
            var predictions = cvResult.Model.Transform(transformedTestData);
            var metrics = context.BinaryClassification.Evaluate(predictions);

            Console.WriteLine($"Accuracy: {metrics.Accuracy}");
            Console.WriteLine($"Area Under Roc Curve: {metrics.AreaUnderRocCurve}");
            Console.WriteLine($"F1 Score: {metrics.F1Score}");
            Console.WriteLine("--------------------------------");
            Console.WriteLine();
        }
    }
}