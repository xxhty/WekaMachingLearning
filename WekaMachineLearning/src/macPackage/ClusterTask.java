package macPackage;

/**
 * Created by TonyHuang on 2/12/17.
 */

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

import weka.core.Instances;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.EM;
public class ClusterTask {
    public static void main(String args[]) throws Exception{

        //load data
        Instances data = new Instances(new BufferedReader(new FileReader("data/bank-data.arff")));

        // new instance of clusterer
        EM model = new EM();
        // build the clusterer
        model.buildClusterer(data);
        System.out.println(model);

        double logLikelihood = ClusterEvaluation.crossValidateModel(model, data, 10, new Random(1));
        System.out.println(logLikelihood);


    }
}
