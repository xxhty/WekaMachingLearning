package KDDCup;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Classifier;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RemoveType;
import weka.filters.unsupervised.attribute.RemoveUseless;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import weka.filters.unsupervised.attribute.Discretize;

import java.io.File;
import java.util.Random;

/**
 * Created by tony.huang on 2/15/2017.
 */
public class main {
    public static void main(String args[])throws Exception {
        {
            Classifier baselineNB = new NaiveBayes();

            double resNB[] = evaluate(baselineNB);
            System.out.println("Naive Bayes\n" +
                    "\tchurn:     " + resNB[0] + "\n" +
                    "\tappetency: " + resNB[1] + "\n" +
                    "\tup-sell:   " + resNB[2] + "\n" +
                    "\toverall:   " + resNB[3] + "\n");
        }

    }
    public static Instances loadData(String pathData, String pathLabeles)
            throws Exception {

		/*
		 * Load data
		 */
        CSVLoader loader = new CSVLoader();
        loader.setFieldSeparator("\t");
        loader.setNominalAttributes("191-last");
        loader.setSource(new File(pathData));
        Instances data = loader.getDataSet();
      //  System.out.println(data);
        // remove String attribute types
        RemoveType removeString = new RemoveType();
        removeString.setOptions(new String[] { "-T", "string" });
        removeString.setInputFormat(data);
        Instances filteredData = Filter.useFilter(data, removeString);
        // data.deleteStringAttributes();

		/*
		 * Load labels
		 */
        CSVLoader loader2 = new CSVLoader();
        loader2.setFieldSeparator("\t");
        loader2.setNoHeaderRowPresent(true);
        loader2.setNominalAttributes("first-last");
        loader2.setSource(new File(pathLabeles));
        Instances labeles = loader2.getDataSet();
        // System.out.println(labeles.toSummaryString());
     //   System.out.println(filteredData.size());
     //   System.out.println(labeles.size());
        // Append label as class value
        Instances labeledData = Instances.mergeInstances(filteredData, labeles);

        // set it as a class value
        labeledData.setClassIndex(labeledData.numAttributes() - 1);

		//System.out.println(labeledData.toSummaryString());

        return labeledData;
	    //test githu
        //test github2

        //test from mac
    }
    public static double[] evaluate(Classifier model) throws Exception {

        double results[] = new double[4];

        String[] labelFiles = new String[] { "churn", "appetency", "upselling" };

        double overallScore = 0.0;
        for (int i = 0; i < labelFiles.length; i++) {

            // Load data
            Instances train_data = loadData("data/orange_small_train.data",
                    "data/orange_small_train_" + labelFiles[i]+ ".labels.txt");
            train_data = preProcessData(train_data);

            // cross-validate the data
            Evaluation eval = new Evaluation(train_data);
            eval.crossValidateModel(model, train_data, 5, new Random(1), new Object[] {});

            // Save results
            results[i] = eval.areaUnderROC(train_data.classAttribute()
                    .indexOfValue("1"));
            Object o=train_data.classAttribute().indexOfValue("1");
            overallScore += results[i];
            System.out.println(labelFiles[i] + "\t-->\t" +results[i]);
        }
        // Get average results over all three problems
        results[3] = overallScore / 3;
        return results;
    }
    public static Instances preProcessData(Instances data) throws Exception{

		/*
		 * Remove useless attributes
		 */
        RemoveUseless removeUseless = new RemoveUseless();
        removeUseless.setOptions(new String[] { "-M", "99" });	// threshold
        removeUseless.setInputFormat(data);
        data = Filter.useFilter(data, removeUseless);


		/*
		 * Remove useless attributes
		 */
        ReplaceMissingValues fixMissing = new ReplaceMissingValues();
        fixMissing.setInputFormat(data);
        data = Filter.useFilter(data, fixMissing);


		/*
		 * Remove useless attributes
		 */
        Discretize discretizeNumeric = new Discretize();
        discretizeNumeric.setOptions(new String[] {
                "-O",
                "-M",  "-1.0",
                "-B",  "4",  // no of bins
                "-R",  "first-last"}); //range of attributes
        fixMissing.setInputFormat(data);
        data = Filter.useFilter(data, fixMissing);

		/*
		 * Select only informative attributes
		 */
        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        Ranker search = new Ranker();
        search.setOptions(new String[] { "-T", "0.001" });	// information gain threshold
        AttributeSelection attSelect = new AttributeSelection();
        attSelect.setEvaluator(eval);
        attSelect.setSearch(search);

        // apply attribute selection
        attSelect.SelectAttributes(data);

        // remove the attributes not selected in the last run
        data = attSelect.reduceDimensionality(data);



        return data;
    }
}
