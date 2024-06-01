using System;
using System.IO;
using System.Linq;

class Program
{
    static void Main()
    {
        // Initialize random generator
        Random rand = new Random();

        // Generate first array of 100,000 random integers
        int[] array1 = new int[100,000];
        for (int i = 0; i < array1.Length; i++)
        {
            array1[i] = rand.Next(0, 46);
        }

        // Generate second array of 100,000 random integers
        int[] array2 = new int[100,000];
        for (int i = 0; i < array2.Length; i++)
        {
            array2[i] = rand.Next(0, 46);
        }

        // Calculate the absolute difference
        int[] absDifference = new int[100,000];
        for (int i = 0; i < absDifference.Length; i++)
        {
            absDifference[i] = Math.Abs(array1[i] - array2[i]);
        }

        // Write the result to a CSV file
        using (StreamWriter writer = new StreamWriter("result.csv"))
        {
            writer.WriteLine("Absolute Difference");
            foreach (int value in absDifference)
            {
                writer.WriteLine(value);
            }
        }

        Console.WriteLine("CSV file has been generated successfully.");
    }
}
