package KDDCup;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.GaussianProcesses;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.pmml.consumer.Regression;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.M5P;
import weka.classifiers.trees.REPTree;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.RemoveType;
import java.io.File;
import java.util.Random;

/**
 * Created by tony.huang on 2/15/2017.
 */
public class main {
    public static void main(String args[])throws Exception {
        {
            double results[] = new double[4];

            String[] labelFiles = new String[] { "churn", "appetency", "upselling" };
            Instances train_data=null;
            double overallScore = 0.0;
            for (int i = 0; i < labelFiles.length; i++) {

                // Load data
                train_data = loadData("data/orange_small_train.data",
                        "data/orange_small_train_" + labelFiles[i] + ".labels.txt");

            }
            //  System.out.println(train_data);

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
        System.out.println(filteredData.size());
        System.out.println(labeles.size());
        // Append label as class value
        Instances labeledData = Instances.mergeInstances(filteredData, labeles);

        // set it as a class value
        labeledData.setClassIndex(labeledData.numAttributes() - 1);

		System.out.println(labeledData.toSummaryString());

        return labeledData;
	    //test githu
        //test github2

        //test from mac
    }

}
