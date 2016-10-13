
import weka.classifiers.Evaluation;
import weka.classifiers.MultipleClassifiersCombiner;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.KStar;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class Main {
	public static void main(String[] args) throws Exception{
		

		Instances data = TextCategorization.readARFF("final.arff");
		data.setClassIndex(data.numAttributes()-1);
	    
	    System.out.println("J48 model");
		J48 j48 = new J48();
		Evaluation eval1 = TextCategorization.generateEvaluation(data,j48, 0.7);
		TextCategorization.printEvaluation(eval1);
		
		
		
		System.out.println("SMO model");
		SMO smo = new SMO();
		Evaluation eval2 = TextCategorization.generateEvaluation(data,smo, 0.7);
		TextCategorization.printEvaluation(eval2);
		
		TextCategorization.serializeClassifier("classifier.model", TextCategorization.generateModel(data, new SMO()) );
		
		System.out.println("NaiveBayes model");
		NaiveBayes naiveBayes = new NaiveBayes();
		Evaluation eval3 = TextCategorization.generateEvaluation(data,naiveBayes, 0.7);
		TextCategorization.printEvaluation(eval3);
		
		System.out.println("MultilayerPerceptron model");
		MultilayerPerceptron mlp = new MultilayerPerceptron();
		Evaluation eval4 = TextCategorization.generateEvaluation(data,mlp, 0.7);
		TextCategorization.printEvaluation(eval4);
		
		System.out.println("Logistic model");
		Logistic logistic = new Logistic();
		Evaluation eval5 = TextCategorization.generateEvaluation(data,logistic, 0.7);
		TextCategorization.printEvaluation(eval5);
		
	}
	
	
	
	
}
