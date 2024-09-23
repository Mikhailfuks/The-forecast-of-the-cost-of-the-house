using System;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace HousePricePrediction
{
    // Define the data class for house features
    public class HouseData
    {
        [LoadColumn(0)]
        public float Size { get; set; } // Size in square feet

        [LoadColumn(1)]
        public float Bedrooms { get; set; } // Number of bedrooms

        [LoadColumn(2)]
        public float Bathrooms { get; set; } // Number of bathrooms

        [LoadColumn(3), ColumnName("Label")]
        public float Price { get; set; } // Price in dollars
    }

    // Define the class for predictions
    public class HousePrediction
    {
        [ColumnName("Score")]
        public float PredictedPrice { get; set; } 
    }

    class Program
    {
        static void Main(string[] args)
        {
            // 1. Load the data
            MLContext mlContext = new MLContext();
            string dataPath = "house_prices.csv"; // Replace with your data file path
            IDataView dataView = mlContext.Data.LoadFromTextFile<HouseData>(dataPath, hasHeader: true, separatorChar: ',');

            // 2. Define the training pipeline
            var pipeline = mlContext.Transforms.Concatenate("Features", "Size", "Bedrooms", "Bathrooms")
                .Append(mlContext.Regression.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features"))
                .Append(mlContext.Transforms.CopyColumns(outputColumnName: "Score", inputColumnName: "PredictedLabel"));

            // 3. Train the model
            ITransformer model = pipeline.Fit(dataView);

            // 4. Create a prediction engine
            var predictionEngine = mlContext.Model.CreatePredictionEngine<HouseData, HousePrediction>(model);

            // 5. Make a prediction
            HouseData newHouse = new HouseData()
            {
                Size = 2500, // Example house size
                Bedrooms = 3, // Example number of bedrooms
                Bathrooms = 2 // Example number of bathrooms
            };

            HousePrediction prediction = predictionEngine.Predict(newHouse);

            // 6. Display the prediction
            Console.WriteLine($"Predicted Price: ${prediction.PredictedPrice}");

            Console.ReadKey();
        }
    }
}
