// See https://aka.ms/new-console-template for more information
using System.Drawing;
using System.Security.Cryptography;

Console.WriteLine("-------Deep Neural Network forward pas------------");

//------Tilføj input her (uden bias, den tilføjes automatisk-------
double[,] input = { { 0, 0, 1 } };
//-----------------------------------------------------------------

//--------Tilføje alle vægte til listen her------------------------
List<double[,]> listeMedAlleVaegte = new List<double[,]>();
listeMedAlleVaegte.Add(
    new double[,]
{{-1.0000,-0.5000,0.1250,0.5000},
{-1.0000,0.0000,-0.5000,0.1111},
{0.5000,-0.1667,-0.1667,0.0000},
{0.1667,-0.1667,-0.1667,0.0000}}
    );
listeMedAlleVaegte.Add(
    new double[,]
{{0.1111,-0.1111},
{0.0000,0.0000},
{0.0000,-0.2500},
{0.1111,-0.2500},
{-1.0000,-0.2500}}
    );

listeMedAlleVaegte.Add(
    new double[,]
{{0.0000,0.2000,0.0000},
{-0.2500,-0.2000,-1.0000},
{0.0000,0.0000,1.0000}}
    );
//-----------------------------------------------------------------

//-------Så køres alt automatisk her-------------------------------
input = MatrixMultiplication.addCollumn(input);
foreach(var matrix in listeMedAlleVaegte)
{
    double[,] ud = MatrixMultiplication.ForwardOneLayer(input, matrix);
    ud = MatrixMultiplication.addCollumn(ud);
    input = ud;
}
//-----------------------------------------------------------------

//------OUTPUT fra systemt er nu gemt i input variablen------------
MatrixMultiplication.PrintMatrix(input);
//-----------------------------------------------------------------



class MatrixMultiplication
{
    static public double[,] ForwardOneLayer(double[,] matrixA, double[,] matrixB)
    {
        bool printAll = false;

        // Get the number of rows and columns
        int rowsA = matrixA.GetLength(0);
        int colsA = matrixA.GetLength(1);
        int rowsB = matrixB.GetLength(0);
        int colsB = matrixB.GetLength(1);

        // Check if multiplication is possible
        if (colsA != rowsB)
        {
            Console.WriteLine("Matrix multiplication not possible.");
            return null;
        }

        // Initialize the result matrix
        double[,] resultMatrix = new double[rowsA, colsB];

        // Perform matrix multiplication
        for (int i = 0; i < rowsA; i++)
        {
            for (int j = 0; j < colsB; j++)
            {
                resultMatrix[i, j] = 0;
                for (int k = 0; k < colsA; k++)
                {
                    resultMatrix[i, j] += matrixA[i, k] * matrixB[k, j];
                }
            }
        }
        // Display the result       
        for (int i = 0; i < rowsA; i++)
        {
            for (int j = 0; j < colsB; j++)
            {
                if(printAll)
                {
                    Console.WriteLine("Sum: " + resultMatrix[i, j] + " ");
                    Console.Write("Sigmoid: " + Sigmoid(resultMatrix[i, j]) + " ");
                    Console.WriteLine();
                }
                resultMatrix[i, j] = Sigmoid(resultMatrix[i, j]);

            }
        }


        return resultMatrix;
    }
    public static double Sigmoid(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }
    public static double[,] addCollumn(double[,] originalArray)
    {
        int originalRows = originalArray.GetLength(0);
        int originalCols = originalArray.GetLength(1);

        // Create a new array with one additional column
        double[,] newArray = new double[originalRows, originalCols + 1];

        // Copy the original array data to the new array
        for (int i = 0; i < originalRows; i++)
        {
            for (int j = 0; j < originalCols; j++)
            {
                newArray[i, j] = originalArray[i, j];
            }
        }
        newArray[0, originalCols] = 1;//Tilføjet bias som hele tiden er 1
        return newArray;
    }
    public static void PrintMatrix(double[,] matrix)
    {       
        for (int i = 0; i < matrix.GetLength(0); i++)
        {
            for (int j = 0; j < matrix.GetLength(1); j++)
            {
                Console.WriteLine( matrix[i, j] + " ");   
            }
        }
    }
}