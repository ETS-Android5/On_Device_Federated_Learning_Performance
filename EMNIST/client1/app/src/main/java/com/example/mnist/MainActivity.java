package com.example.mnist;

import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.net.Socket;
import java.util.Random;

import au.com.bytecode.opencsv.CSVWriter;

//import android.support.v7.app.AppCompatActivity;


public class MainActivity extends AppCompatActivity {
    private static final String basePath = Environment.getExternalStorageDirectory() + "/mnist";
    private static final String mnistTrainUrl = "https://github.com/KANG-FU/On_device_FL_dataset/raw/main/mnist_client1_non_iid.tar.gz";
    private static final String clientID = "1";
    // the I/O stream for sending and receiving the model
    private static DataInputStream din;
    private static DataOutputStream dout;
    double trainTime;
    TextView textTime;
    TextView textAccuracy;
    static String serverIP = "192.168.137.103";
    String trainDataset = "client1_mnist_iid_batch";
            
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button button = (Button) findViewById(R.id.button);
        textTime = (TextView) findViewById(R.id.textView2);
        textAccuracy = (TextView) findViewById(R.id.textAccuracy);

        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                AsyncTaskRunner runner = new AsyncTaskRunner();
                runner.execute("");
                ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
                bar.setVisibility(View.VISIBLE);
            }
        });
    }

    private class AsyncTaskRunner extends AsyncTask<String, Integer, String> {


        // Runs in UI before background thread is called
        @Override
        protected void onPreExecute() {
            super.onPreExecute();
            ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
            bar.setVisibility(View.INVISIBLE);
            textTime.setText("Start training...");
        }

        // This is our main background thread for training the model and uploading the model
        @Override
        protected String doInBackground(String... params) {
            try {
                // download the dataset from the Internet
                if (!new File(basePath + "/mnist_client1_iid").exists()) {
                    Log.d("Data download", "Data downloaded from " + mnistTrainUrl);
                    File modelDir = new File(basePath + "/mnist_client1_iid");
                    if (!modelDir.exists()) {
                        modelDir.mkdirs();
                    }
                    if (DataUtilities.downloadFile(mnistTrainUrl, basePath)) {
                        DataUtilities.extractTarGz(basePath+"/mnist_client1_iid.tar.gz", basePath + "/mnist_client1_iid");
                    }
                }
                // the beginning timestamp
                double beginTime = System.nanoTime();

                // write the training time and current absolute time to csv of each round
                File file = new File(basePath + "/"+ trainDataset + ".csv");
                FileWriter output = null;
                try {
                    output = new FileWriter(file);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                CSVWriter write = new CSVWriter(output);
                // Header column value
                String[] header = { "ID", "Training Time", "Current Time" };
                int num = 0;
                write.writeNext(header);

                // load the training dataset
                modelTrain model = new modelTrain();
                DataSetIterator mnistTrain = model.loadTrainData();

                // The training process
                // First it will receive the model from the server and then load the model to train it using its own dataset
                // After that, send the trained model back to the server and waiting for the new model from the server
                while(true){
                    receiveModel();
                    String updatedModelPath = basePath + "/updatedModel.zip";
                    MultiLayerNetwork modelLoad = ModelSerializer.restoreMultiLayerNetwork(updatedModelPath);
                    trainTime = model.modelTrain(modelLoad,mnistTrain);
                    sendModel();
                    num = num + 1;
                    if (receiveSignal() == 1){
                        Thread.sleep(5000);
                        double currentTime = (System.nanoTime()-beginTime)/1000000000;
                        String[] data={String.valueOf(num),String.valueOf(trainTime), String.valueOf(currentTime)};
                        write.writeNext(data);
                        continue;
                    }
                    else{
                        double currentTime = (System.nanoTime()-beginTime)/1000000000;
                        String[] data={String.valueOf(num),String.valueOf(trainTime), String.valueOf(currentTime)};
                        write.writeNext(data);
                        write.close();
                        break;
                    }
                }

            }catch(Exception e){
                e.printStackTrace();
            }
            return "";
        }

        //This is called from background thread but runs in UI for a progress indicator
        @Override
        protected void onProgressUpdate(Integer... values) {
            super.onProgressUpdate(values);
        }

        //This block executes in UI when background thread finishes
        //This is where we update the UI with our classification results
        @Override
        protected void onPostExecute(String result) {
            super.onPostExecute(result);
            //Hide the progress bar now that we are finished
            ProgressBar bar = (ProgressBar) findViewById(R.id.progressBar);
            bar.setVisibility(View.INVISIBLE);
            textTime.setText("The time for training is "+ trainTime + "s");
        }

    }

    // The class for train the model
    // It has two methods. One is load the training data and the other is train the model.
    private  class modelTrain {
        final int numRows = 28;
        final int numColumns = 28;
        int channels = 1; // single channel for grayscale images
        int outputNum = 10; // number of output classes
        int batchSize = 64; // batch size for each epoch
        int rngSeed = 1234; // random number seed for reproducibility
        Random randNumGen = new Random(rngSeed);
        int numEpochs = 1; // number of epochs to perform
        int numBatch = 25;

        private DataSetIterator loadTrainData() throws IOException, InterruptedException {

            // vectorization of train data
            File trainData = new File(basePath + "/emnist_client1_non_iid");
            FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
            ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator(); // parent path as the image label
            ImageRecordReader trainRR = new ImageRecordReader(numRows, numColumns, channels, labelMaker);
            trainRR.initialize(trainSplit);
            final DataSetIterator mnistTrain = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);
            // transform pixel values from 0-255 to 0-1 (min-max scaling)
            DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
            scaler.fit(mnistTrain);
            mnistTrain.setPreProcessor(scaler);

            return mnistTrain;
        }



        private double modelTrain(MultiLayerNetwork myNetwork, DataSetIterator mnistTrain) throws IOException {
            Log.d("train model", "Train model....");

            double startTime = System.nanoTime();
            for (int l = 0; l < numBatch; l++) {
                DataSet ds = mnistTrain.next();
                myNetwork.fit(ds);
            }
//            myNetwork.fit(mnistTrain, numEpochs);
            trainTime = (System.nanoTime()-startTime)/1000000000;
            System.out.println("The time for training is "+ trainTime + "s");
            ModelSerializer.writeModel(myNetwork,  new File(basePath+"/localModel_cID_" + clientID +".zip" ), true);
            return trainTime;
        }

    }

    // This is function for receiving the model from the server
    private static void receiveModel() throws IOException {
        System.out.println("Connecting....");
        Socket socket = new Socket(serverIP, 5000);
        System.out.println("Connected!");

        din = new DataInputStream(socket.getInputStream());
        dout = new DataOutputStream(socket.getOutputStream());
        System.out.println("Receiving model from server...");
        receiveFile();
        System.out.println("Model Received!");

        System.out.println("Closing socket.");
        socket.close();
    }

    // This is function for sending the model to the server
    private static void sendModel() throws IOException {
        System.out.println("Connecting....");
        Socket socket = new Socket(serverIP, 5000);
        System.out.println("Connected!");

        din = new DataInputStream(socket.getInputStream());
        dout = new DataOutputStream(socket.getOutputStream());

        File fileSent = new File(basePath+"/localModel_cID_" + clientID +".zip" );
        System.out.println("Trained Model sending to server...");
        sendFile_from_client(fileSent);			//the file if present, is sent over the network
        System.out.println("Trained Model sent!");
        System.out.println(din.readUTF());
        System.out.println("Closing socket and terminating program.");;
        socket.close();
    }

    // This is the function to know if the server asks the clients to stop training as the model has reached the target accuracy
    private static int receiveSignal() throws IOException {
        System.out.println("Connecting....");
        Socket socket = new Socket(serverIP, 5000);
        System.out.println("Connected!");

        din = new DataInputStream(socket.getInputStream());
        dout = new DataOutputStream(socket.getOutputStream());
        System.out.println("Receiving signal from server to decide if continuing training...");
        int signal = din.readInt();

        socket.close();
        return signal ;
    }

    // The function for sending files using sockets
    private static void sendFile_from_client(File file) {
        try {
            dout.writeUTF(file.getName());
            //creating byteArray with length same as file length
            byte[] byteArray = new byte[(int) file.length()];
            dout.writeInt(byteArray.length);
            BufferedInputStream bis = new BufferedInputStream (new FileInputStream(file));
            //Writing int 0 as a Flag which denotes the file is present in the Server directory, if file was absent, FileNotFound exception will be thrown and int 1 will be written
            dout.writeInt(0);
            BufferedOutputStream bos = new BufferedOutputStream(dout);
            int count;
            while((count = bis.read(byteArray)) != -1) {			//reads bytes of byteArray length from the BufferedInputStream into byteArray
                //writes bytes from byteArray into the BufferedOutputStream (0 is the offset and count is the length)
                bos.write(byteArray, 0, count);
            }
            bos.flush();
            bis.close();
            //readInt is used to reset if any bytes are present in the buffer after the file transfer
            din.readInt();
        }
        catch(FileNotFoundException ex) {
            System.out.println("File "  + " Not Found! \n        Please Check the input and try again.\n\n        ");
            try {
                //Writing int 1 as a Flag which denotes the file is absent from the Server directory, if file was present int 0 would be written
                dout.writeInt(1);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        catch(IOException ex) {
            ex.printStackTrace();
        }
    }

    // The function for receiving files using sockets
    private static void receiveFile() {
        int bytesRead = 0, current = 0;
        try {
            int fileLength = din.readInt();
            //creating byteArray with length same as file length
            byte[] byteArray = new byte[fileLength];
            BufferedInputStream bis = new BufferedInputStream(din);
            File file = new File(basePath+"/updatedModel.zip");
            //fileFoundFlag is a Flag which denotes the file is present or absent from the Server directory, is present int 0 is sent, else 1
            int fileFoundFlag = din.readInt();
            if(fileFoundFlag == 1)
                return;
            BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(file));
            //reads bytes of length byteArray from BufferedInputStream & writes into the byteArray, (Offset 0 and length is of byteArray)
            bytesRead = bis.read(byteArray, 0, byteArray.length);
            current = bytesRead;
            //Sometimes only a portion of the file is read, hence to read the remaining portion...
            do {
                //BufferedInputStream is read again into the byteArray, offset is current (which is the amount of bytes read previously) and length is the empty space in the byteArray after current is subtracted from its length
                bytesRead = bis.read(byteArray, current, (byteArray.length - current));

                if(bytesRead >= 0)
                    current += bytesRead;					//current is updated after the new bytes are read
            } while(bytesRead > 0);
            //writes bytes from the byteArray into the BufferedOutputStream, offset is 0 and length is current (which is the amount of bytes read into byteArray)
            bos.write(byteArray, 0, current);
            bos.close();
            System.out.println("Model Successfully Downloaded!" );
            //writeInt is used to reset if any bytes are present in the buffer after the file transfer
            dout.writeInt(0);
        }
        catch(IOException ex) {
            ex.printStackTrace();
        }
    }
}