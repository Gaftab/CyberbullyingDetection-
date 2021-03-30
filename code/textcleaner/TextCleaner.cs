using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;

namespace DataMergingApp
{
    public static class TextCleaner
    {

        /// <summary>
        /// Return cleaned version of text after some operations.
        /// </summary>
        /// <param name="text"></param>
        /// <returns></returns>
        public static string CleanText(string text)
        {
            text = Regex.Replace(text, @"\t|\n|\r", " ");
            text = Regex.Replace(text, @"http[^\s]+", "");
            char[] arr = text.ToCharArray();
            arr = Array.FindAll<char>(arr, (c => (char.IsLetterOrDigit(c) || char.IsWhiteSpace(c))));
            string cleanedText = new string(arr);
            cleanedText.TrimStart(' ');
            return cleanedText.ToLower();
        }


        static List<string> stopWords;
        static List<string> slangWords;

        /// <summary>
        /// Corrects the  a word if it is misspelled slang word
        /// </summary>
        /// <param name="word"></param>
        /// <returns></returns>
        public static bool IsSlangWord(ref string word)
        {
            string temp = "";
            int counter = 1;
            if (string.IsNullOrEmpty(word))
                return false;
            char prevChar = word[0];
            temp += prevChar;
            for (int j = 1; j < word.Length; j++)
            {
                if (prevChar == word[j])
                {
                    counter++;
                    if (counter > 2)
                    {
                        continue;
                    }
                    else
                    {
                        temp += word[j];
                    }
                }
                else
                {
                    temp += word[j];
                    prevChar = word[j];
                    counter = 1;
                }
            }

            if (temp != word)
            {
                //editedWords += word + "-" + temp + "\n";
                word = string.Copy(temp);
            }


            foreach (string slangWord in slangWords)
            {
                if (string.Equals(slangWord, word)) return true;
                else if (slangWord.Length > 3 && LevenshteinSimilarity(word.Substring(0, Math.Min(word.Length, slangWord.Length)), slangWord) > 0.82)
                {
                    //editedWords += word + "-" + slangWord + "\n";
                    word = slangWord;
                    return true;
                }
                else if (slangWord.Length > 3 && word.Contains(slangWord)) return true;
            }
            return false;
        }

        public static bool IsStopWord(string word)
        {
            return stopWords.Contains(word);
        }

        public static void SetSlangWords()
        {
            slangWords = new List<string>();
            using (StreamReader file = new StreamReader(@"/data/turkish_slangwords.txt"))
            {
                string word;
                while ((word = file.ReadLine()) != null)
                {
                    slangWords.Add(word);
                }
            }
        }

        public static void SetStopWords()
        {
            stopWords = new List<string>();
            using (StreamReader file = new StreamReader(@"/data/turkish_stopwords.txt"))
            {
                string word;
                while ((word = file.ReadLine()) != null)
                {
                    stopWords.Add(word);
                }
            }
        }

        /// <summary>
        ///  Returns the number of steps required to transform the source string
        /// </summary>
        /// <param name="source"></param>
        /// <param name="target"></param>
        /// <returns></returns>
        public static int LevenshteinDistance(string source, string target)
        {
            if ((source == null) || (target == null)) return 0;
            if ((source.Length == 0) || (target.Length == 0)) return 0;
            if (source == target) return source.Length;

            int sourceWordCount = source.Length;
            int targetWordCount = target.Length;

            if (sourceWordCount == 0)
                return targetWordCount;

            if (targetWordCount == 0)
                return sourceWordCount;

            int[,] distance = new int[sourceWordCount + 1, targetWordCount + 1];

            for (int i = 0; i <= sourceWordCount; distance[i, 0] = i++) ;
            for (int j = 0; j <= targetWordCount; distance[0, j] = j++) ;

            for (int i = 1; i <= sourceWordCount; i++)
            {
                for (int j = 1; j <= targetWordCount; j++)
                {
                    int cost = (target[j - 1] == source[i - 1]) ? 0 : 1;
                    distance[i, j] = Math.Min(Math.Min(distance[i - 1, j] + 1, distance[i, j - 1] + 1), distance[i - 1, j - 1] + cost);
                }
            }

            return distance[sourceWordCount, targetWordCount];
        }

        /// <summary>
        /// Calculate percentage similarity of two strings with respect to Levenshtein
        /// </summary>
        /// <param name="source"></param>
        /// <param name="target"></param>
        /// <returns></returns>
        public static double LevenshteinSimilarity(string source, string target)
        {
            if ((source == null) || (target == null)) return 0.0;
            if ((source.Length == 0) || (target.Length == 0)) return 0.0;
            if (source == target) return 1.0;

            int stepsToSame = LevenshteinDistance(source, target);
            return (1.0 - ((double)stepsToSame / (double)Math.Max(source.Length, target.Length)));
        }

  
    }
}
