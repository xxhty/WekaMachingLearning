package macPackage;
import weka.classifiers.Classifier;

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

import java.io.File;
import java.util.Random;

/**
 * Created by TonyHuang on 2/12/17.
 */
public class RegressionTask  {
    public static void main(String[] args) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setFieldSeparator(",");
        loader.setSource(new File("data/ENB2012_data.csv"));
        Instances data = loader.getDataSet();

        Remove remove = new Remove();
        remove.setOptions(new String[] { "-R", data.numAttributes() + "" });
        remove.setInputFormat(data);
        data = Filter.useFilter(data, remove);
        System.out.println(data);

        data.setClassIndex(data.numAttributes()-1);
        LinearRegression model=new LinearRegression();
        model.buildClassifier(data);
        System.out.println(model);


        weka.classifiers.Evaluation eval=new weka.classifiers.Evaluation(data);
        eval.crossValidateModel(model,data,20,new Random(1),new String[] {});
        System.out.println(eval.toSummaryString());

        M5P md5=new M5P();
        md5.setOptions(new String[] {});
        md5.buildClassifier(data);
        System.out.println(md5);

        // 10-fold cross-validation
        eval.crossValidateModel(md5, data, 10, new Random(1), new String[] {});
        System.out.println(eval.toSummaryString());
        System.out.println();

        /*
		 * Bonus: Build additional models
		 */

         ZeroR modelZero = new ZeroR();



        System.out.println("--------------------------------");

         REPTree modelTree = new REPTree();
         modelTree.buildClassifier(data);
         System.out.println(modelTree);
         eval = new Evaluation(data);
         eval.crossValidateModel(modelTree, data, 10, new Random(1), new
         String[]{});
         System.out.println(eval.toSummaryString());

         SMOreg modelSVM = new SMOreg();

         MultilayerPerceptron modelPerc = new MultilayerPerceptron();

         GaussianProcesses modelGP = new GaussianProcesses();
         modelGP.buildClassifier(data);
         System.out.println(modelGP);
         eval = new Evaluation(data);
         eval.crossValidateModel(modelGP, data, 10, new Random(1), new
         String[]{});
         System.out.println(eval.toSummaryString());
    }

}
