package com.journaldev.androidcameraxopencv;

import static org.opencv.core.Core.NORM_L2;
import static org.opencv.core.Core.NORM_MINMAX;
import static org.opencv.core.Core.bitwise_not;
import static org.opencv.core.Core.divide;
import static org.opencv.core.Core.findNonZero;
import static org.opencv.core.Core.inRange;
import static org.opencv.core.Core.magnitude;
import static org.opencv.core.Core.mean;
import static org.opencv.core.Core.minMaxLoc;
import static org.opencv.core.Core.multiply;
import static org.opencv.core.Core.norm;
import static org.opencv.core.Core.normalize;
import static org.opencv.core.Core.perspectiveTransform;
import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_64F;
import static org.opencv.core.CvType.CV_64FC2;
import static org.opencv.core.CvType.CV_8U;
import static org.opencv.core.Mat.ones;
import static org.opencv.core.Mat.zeros;
import static org.opencv.core.TermCriteria.COUNT;
import static org.opencv.core.TermCriteria.EPS;
import static org.opencv.core.TermCriteria.MAX_ITER;
import static org.opencv.features2d.Features2d.drawKeypoints;
import static org.opencv.imgproc.Imgproc.CHAIN_APPROX_SIMPLE;
import static org.opencv.imgproc.Imgproc.COLOR_BGR2GRAY;
import static org.opencv.imgproc.Imgproc.RETR_LIST;
import static org.opencv.imgproc.Imgproc.*;
import static org.opencv.calib3d.Calib3d.*;
import static org.opencv.imgproc.Imgproc.cvtColor;
import static org.opencv.imgproc.Imgproc.drawContours;
import static org.opencv.imgproc.Imgproc.ellipse;
import static org.opencv.imgproc.Imgproc.findContours;
import static org.opencv.imgproc.Imgproc.fitEllipse;

import static java.lang.Math.PI;
import static java.lang.Math.abs;
import static java.lang.Math.atan;
import static java.lang.Math.cos;
import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.Math.pow;
import static java.lang.Math.sin;
import static java.lang.Math.sqrt;
import static java.lang.Math.tan;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraX;
import androidx.camera.core.Exif;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageAnalysisConfig;
import androidx.camera.core.ImageCapture;
import androidx.camera.core.ImageCaptureConfig;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.PreviewConfig;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PointF;
import android.media.ExifInterface;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Rational;
import android.util.Size;
import android.view.Menu;
import android.view.MenuItem;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.Toast;
import com.google.android.material.floatingactionbutton.FloatingActionButton;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.TermCriteria;
import org.opencv.features2d.SimpleBlobDetector;
import org.opencv.features2d.SimpleBlobDetector_Params;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Vector;


public class MainActivity extends AppCompatActivity implements View.OnClickListener {


    double CV_PI = 3.1415926535897932384626433832795;
    private int REQUEST_CODE_PERMISSIONS = 101;
    private final String[] REQUIRED_PERMISSIONS = new String[]{"android.permission.CAMERA", "android.permission.WRITE_EXTERNAL_STORAGE"};
    TextureView textureView;
    ImageView ivBitmap;
    LinearLayout llBottom;
    Scalar COLOR_RED = new Scalar(255.0, 0.0, 0.0, 255.0);
    Scalar COLOR_GREEN = new Scalar(0.0, 255.0, 0.0, 255.0);
    Scalar COLOR_BLUE = new Scalar(0.0, 0.0, 255.0, 255.0);
    Scalar COLOR_YELLOW = new Scalar(255.0, 255.0, 0.0, 255.0);
    org.opencv.core.Size IMAGE_SIZE = new org.opencv.core.Size(1512, 2016);
    //org.opencv.core.Size MAX_CHESSBOARD_SIZE = new org.opencv.core.Size(9,6);
    org.opencv.core.Size chessboardSize = new org.opencv.core.Size(8,6);
    org.opencv.core.Size circleGridSize = new org.opencv.core.Size(6,8);
    boolean CAN_TAKE_PHOTO = false;

    String currentImageProcessing = "CHESSBOARD";

    ImageCapture imageCapture;
    ImageAnalysis imageAnalysis;
    Preview preview;

    FloatingActionButton btnCapture, btnOk, btnCancel;

    static {
        if (!OpenCVLoader.initDebug())
            Log.d("ERROR", "Unable to load OpenCV");
        else
            Log.d("SUCCESS", "OpenCV loaded");
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btnCapture = findViewById(R.id.btnCapture);
        btnOk = findViewById(R.id.btnAccept);
        btnCancel = findViewById(R.id.btnReject);

        btnOk.setOnClickListener(this);
        btnCancel.setOnClickListener(this);

        llBottom = findViewById(R.id.llBottom);
        textureView = findViewById(R.id.textureView);
        ivBitmap = findViewById(R.id.ivBitmap);

        if (allPermissionsGranted()) {
            startCamera();
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS);
        }
    }

    private void startCamera() {

        CameraX.unbindAll();

        preview = setPreview();
        imageCapture = setImageCapture();
        imageAnalysis = setImageAnalysis();

        //bind to lifecycle:
        CameraX.bindToLifecycle(this, preview, imageCapture, imageAnalysis);
    }


    private Preview setPreview() {

        Rational aspectRatio = new Rational(textureView.getWidth(), textureView.getHeight());
        Size screen = new Size(textureView.getWidth(), textureView.getHeight()); //size of the screen

        PreviewConfig pConfig = new PreviewConfig.Builder().setTargetAspectRatio(aspectRatio).setTargetResolution(screen).build();
        Preview preview = new Preview(pConfig);

        preview.setOnPreviewOutputUpdateListener(
                new Preview.OnPreviewOutputUpdateListener() {
                    @Override
                    public void onUpdated(Preview.PreviewOutput output) {
                        ViewGroup parent = (ViewGroup) textureView.getParent();
                        parent.removeView(textureView);
                        parent.addView(textureView, 0);

                        textureView.setSurfaceTexture(output.getSurfaceTexture());
//                        updateTransform();
                    }
                });

        return preview;
    }


    private ImageCapture setImageCapture() {
        ImageCaptureConfig imageCaptureConfig = new ImageCaptureConfig.Builder().setCaptureMode(ImageCapture.CaptureMode.MIN_LATENCY)
                .setTargetRotation(Surface.ROTATION_0).build();
        final ImageCapture imgCapture = new ImageCapture(imageCaptureConfig);


        btnCapture.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                if(currentImageProcessing.equals("CCTAG")) {
                    /*imgCapture.takePicture(new ImageCapture.OnImageCapturedListener() {
                        @Override
                        public void onCaptureSuccess(ImageProxy image, int rotationDegrees) {
                            Bitmap bitmap = textureView.getBitmap();
                            showAcceptedRejectedButton(true);
                            ivBitmap.setImageBitmap(bitmap);
                        }

                        @Override
                        public void onError(ImageCapture.UseCaseError useCaseError, String message, @Nullable Throwable cause) {
                            super.onError(useCaseError, message, cause);
                        }
                    });*/
                }
                if(currentImageProcessing.equals("ASSYMETRIC_CIRCLES")) {
                    imgCapture.takePicture(new ImageCapture.OnImageCapturedListener() {
                        @Override
                        public void onCaptureSuccess(ImageProxy image, int rotationDegrees) {
                            Bitmap bitmap = textureView.getBitmap();
                            showAcceptedRejectedButton(true);
                            ivBitmap.setImageBitmap(bitmap);
                        }

                        @Override
                        public void onError(ImageCapture.UseCaseError useCaseError, String message, @Nullable Throwable cause) {
                            super.onError(useCaseError, message, cause);
                        }
                    });
                }


                File file = new File(
                        Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "" + System.currentTimeMillis() + "_JDCameraX.jpg");
                imgCapture.takePicture(file, new ImageCapture.OnImageSavedListener() {
                    @Override
                    public void onImageSaved(@NonNull File file) {
                        Bitmap bitmap = textureView.getBitmap();
                        showAcceptedRejectedButton(true);
                        ivBitmap.setImageBitmap(bitmap);
                    }

                    @Override
                    public void onError(@NonNull ImageCapture.UseCaseError useCaseError, @NonNull String message, @Nullable Throwable cause) {

                    }
                });
            }
        });

        return imgCapture;
    }


    private ImageAnalysis setImageAnalysis() {

        // Setup image analysis pipeline that computes average pixel luminance
        HandlerThread analyzerThread = new HandlerThread("OpenCVAnalysis");
        analyzerThread.start();

        ImageAnalysisConfig imageAnalysisConfig = new ImageAnalysisConfig.Builder()
                .setImageReaderMode(ImageAnalysis.ImageReaderMode.ACQUIRE_LATEST_IMAGE)
                .setCallbackHandler(new Handler(analyzerThread.getLooper()))
                .setImageQueueDepth(1).build();

        ImageAnalysis imageAnalysis = new ImageAnalysis(imageAnalysisConfig);

        imageAnalysis.setAnalyzer(
                new ImageAnalysis.Analyzer() {
                    @Override
                    public void analyze(ImageProxy image, int rotationDegrees) {
                        //Analyzing live camera feed begins.

                        final Bitmap bitmap = textureView.getBitmap();

//                        final Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.template_photo);

                        if (bitmap == null)
                            return;

                        Mat matColor = new Mat();
                        Utils.bitmapToMat(bitmap, matColor);

//---------------------------------------------------------------------------
                        if (currentImageProcessing.equals("CHESSBOARD")) {
//                            matColor = getChessboardCorners(bitmap);
                            List<Point> imagePoints = getChessboardCorners(bitmap).toList();
                            for(Point corner : imagePoints) {
                                drawMarker(matColor, corner, COLOR_RED, 1, 2, 2, 1);
                                putText(matColor, String.valueOf(imagePoints.indexOf(corner)), corner, FONT_HERSHEY_SIMPLEX, 1, COLOR_RED);
                            }
//                            putText(matColor, "corners = " + imagePoints.size(), new Point(100, 100), FONT_HERSHEY_SIMPLEX, 2, COLOR_GREEN);
//                            List<Point> pointsList = new ArrayList<>();
//                            for (int y = 0; y < 6; ++y) {
//                                for (int x = 0; x < 8; ++x) {
//                                    Point row = new Point(100 * x + 100, 100 * y + 100);
//                                    pointsList.add(row);
//                                }
//                            }
//
//                            for (Point p : pointsList) {
//                                putText(matColor, String.valueOf(pointsList.indexOf(p)), p, FONT_HERSHEY_SIMPLEX, 1, COLOR_RED);
//                            }

//                            MatOfPoint2f corners = getChessboardCorners(bitmap);

//                            if(corners.toList().size() >= 4) {
//                                drawMarker(matColor, new Point(corners.get(0, 0)), COLOR_RED, 1, 10, 5, 1);
//                                drawMarker(matColor, new Point(corners.get((int) chessboardSize.width - 1, 0)), COLOR_BLUE, 1, 10, 5, 1);
//                                drawMarker(matColor, new Point(corners.get((int) (chessboardSize.width * (chessboardSize.height - 1)), 0)), COLOR_GREEN, 1, 10, 5, 1);
//                                drawMarker(matColor, new Point(corners.get((int) (chessboardSize.width * chessboardSize.height) - 1, 0)), COLOR_YELLOW, 1, 10, 5, 1);
//                            }
                        } else if (currentImageProcessing.equals("CCTAG")) {
//                            matColor = detectCcTags(bitmap);
                            List<Point> imagePoints = detectCcTags(bitmap).toList();
                            for (Point point : imagePoints) {
                                drawMarker(matColor, point, COLOR_RED, 1, 2, 2, 1);
                                putText(matColor, "" + imagePoints.indexOf(point), point, FONT_HERSHEY_SIMPLEX, 0.8, COLOR_RED, 1);
                            }

                        } else if (currentImageProcessing.equals("ASSYMETRIC_CIRCLES")) {
//                            matColor = getAssymetricCircleCenters(bitmap);
                            List<Point> imagePoints = getAssymetricCircleCenters(bitmap).toList();
                            for (Point point : imagePoints) {
                                drawMarker(matColor, point, COLOR_RED, 1, 2, 2, 1);
                                putText(matColor, String.valueOf(imagePoints.indexOf(point)), point, FONT_HERSHEY_SIMPLEX, 1, COLOR_RED);
                            }
                        }
//---------------------------------------------------------------------------

                        Utils.matToBitmap(matColor, bitmap);
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                ivBitmap.setImageBitmap(bitmap);
                            }
                        });

                    }
                });

        return imageAnalysis;
    }

    private void clearDirectory() {
        Log.d("Debug:", "Deleting Photos");

        String[] children = getFilesDir().list();

        for (String child : children) {
            File file = new File(getFilesDir() + "/" + child);
            Log.d("Debug:", String.valueOf(file));
            file.delete();
        }
    }

    private void calibration() {

        MatOfPoint3f objectPoints = new MatOfPoint3f();
        List<Point3> pointsList = new ArrayList<>();

        Mat matTest = new Mat();
        Mat matTest2 = new Mat();

        if(currentImageProcessing.equals("CCTAG")) {
            Bitmap bitmapRef = BitmapFactory.decodeResource(getResources(), R.drawable.cctagref);
            Utils.bitmapToMat(bitmapRef, matTest);
            int j = 0;
            for(Point imagePoint : detectCcTags(bitmapRef).toArray()) {
                pointsList.add(new Point3(imagePoint.x, imagePoint.y, 0));
                drawMarker(matTest, imagePoint, COLOR_RED, 1, 2, 2, 1);
                putText(matTest, "" + detectCcTags(bitmapRef).toList().indexOf(imagePoint), imagePoint, FONT_HERSHEY_SIMPLEX, 0.8, COLOR_RED, 1);
                j++;
                if (pointsList.size() == 48) {
                    break;
                }
            }

            pointsList.remove(47);
            pointsList.remove(46);
            pointsList.remove(45);
            pointsList.remove(44);
            pointsList.remove(43);
            pointsList.remove(42);
            pointsList.remove(41);
            pointsList.remove(40);
//
            pointsList.remove(39);
//
            pointsList.remove(38);
            pointsList.remove(37);
            pointsList.remove(36);
            pointsList.remove(35);
            pointsList.remove(34);
            pointsList.remove(33);
//
            pointsList.remove(32);

            objectPoints.fromList(pointsList);



            j = 0;

            File fileDir = getFilesDir();
            List<Mat> photosObjectPoints = new ArrayList<>();
            List<Mat> photosImagePoints = new ArrayList<>();
            MatOfPoint2f photoImagePoints2f = new MatOfPoint2f();
            List<Point> imagePointsList2f;

            for (int i = 0; i < fileDir.listFiles().length; i++) {
                photosObjectPoints.add(objectPoints);
                imagePointsList2f = new ArrayList<>(detectCcTags(BitmapFactory.decodeFile(fileDir.listFiles()[i].getPath())).toList());

                imagePointsList2f.remove(47);
                imagePointsList2f.remove(46);
                imagePointsList2f.remove(45);
                imagePointsList2f.remove(44);
                imagePointsList2f.remove(43);
                imagePointsList2f.remove(42);
                imagePointsList2f.remove(41);
                imagePointsList2f.remove(40);
//
                imagePointsList2f.remove(39);
//
                imagePointsList2f.remove(38);
                imagePointsList2f.remove(37);
                imagePointsList2f.remove(36);
                imagePointsList2f.remove(35);
                imagePointsList2f.remove(34);
                imagePointsList2f.remove(33);
//
                imagePointsList2f.remove(32);

                photoImagePoints2f.fromList(imagePointsList2f);
//                Utils.bitmapToMat(BitmapFactory.decodeFile(fileDir.listFiles()[i].getPath()), matTest2);

//                for(Point imagePoint : detectCcTags(BitmapFactory.decodeFile(fileDir.listFiles()[i].getPath())).toList()) {
//                    //pointsList.add(new Point3(imagePoint.x, imagePoint.y, 0));
//                    circle(matTest2, new Point(imagePoint.x, imagePoint.y), 3, COLOR_RED, -1);
//                    putText(matTest2, String.valueOf(j), imagePoint,
//                            FONT_HERSHEY_SIMPLEX, 1, COLOR_BLUE, 3);
//                    j++;
//                }

                Log.d("Debug:", "" + photoImagePoints2f.size() + fileDir.listFiles()[i].getPath());
                photosImagePoints.add(photoImagePoints2f);
            }

            Mat cameraMatrix = new Mat();
            MatOfDouble distCoeffs = new MatOfDouble();
            List<Mat> rvecs = new ArrayList<>();
            List<Mat> tvecs = new ArrayList<>();

            calibrateCamera(photosObjectPoints, photosImagePoints, IMAGE_SIZE, cameraMatrix, distCoeffs, rvecs, tvecs);

            Bitmap bitmap = BitmapFactory.decodeFile(fileDir.listFiles()[4].getPath());
            Mat matDistorted = new Mat();
            Mat matUndistorted = new Mat();
            Utils.bitmapToMat(bitmap, matDistorted);
            double h = matDistorted.size().height;
            double w = matDistorted.size().width;

            Mat newCameraMat = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, new org.opencv.core.Size(w, h), 1);

            undistort(matDistorted, matUndistorted, cameraMatrix, distCoeffs, newCameraMat);

            double mean_error = 0;
            MatOfPoint2f imgpoints2 = new MatOfPoint2f();
            for(int i = 0; i < photosObjectPoints.size(); i++) {
                projectPoints((MatOfPoint3f) photosObjectPoints.get(i), rvecs.get(i), tvecs.get(i), cameraMatrix, distCoeffs, imgpoints2);
                double error = norm(photosImagePoints.get(i), imgpoints2, NORM_L2) / imgpoints2.toList().size();
                mean_error += error;
            }
            putText(matUndistorted, "reprojection error: " + mean_error, new Point(100, 100),
                    FONT_HERSHEY_SIMPLEX, 1, COLOR_BLUE, 3);

            Utils.matToBitmap(matUndistorted, bitmap);
            File file2 = new File(getFilesDir(), "Calibrated.jpg");
            try {
                FileOutputStream out = new FileOutputStream(file2);
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
                out.flush();
                out.close();
            } catch (Exception e) {
            }

        } else if(currentImageProcessing.equals("ASSYMETRIC_CIRCLES")) {
            for (int y = 0; y < circleGridSize.height; ++y) {
                for (int x = 0; x < circleGridSize.width; ++x) {
                    Point3 row;
                    if(y % 2 == 0) {
                        row = new Point3(x * 2, y, 0);
                    } else {
                        row = new Point3((x * 2) + 1, y, 0);
                    }
                    pointsList.add(row);
                }
            }

            pointsList.remove(47);
            pointsList.remove(46);
            pointsList.remove(45);
            pointsList.remove(44);
            pointsList.remove(43);
            pointsList.remove(42);
            pointsList.remove(41);
            pointsList.remove(40);
//
            pointsList.remove(39);
//
            pointsList.remove(38);
            pointsList.remove(37);
            pointsList.remove(36);
            pointsList.remove(35);
            pointsList.remove(34);
            pointsList.remove(33);
//
            pointsList.remove(32);

            objectPoints.fromList(pointsList);

            File fileDir = getFilesDir();
            List<Mat> photosObjectPoints = new ArrayList<>();
            List<Mat> photosImagePoints = new ArrayList<>();
            Mat photoImagePoints;
            MatOfPoint2f photoImagePoints2f = new MatOfPoint2f();
            List<Point> imagePointsList2f = new ArrayList<>();

            for (int i = 0; i < fileDir.listFiles().length; i++) {
                photosObjectPoints.add(objectPoints);
                imagePointsList2f = new ArrayList<>(getAssymetricCircleCenters(BitmapFactory.decodeFile(fileDir.listFiles()[i].getPath())).toList());

                imagePointsList2f.remove(47);
                imagePointsList2f.remove(46);
                imagePointsList2f.remove(45);
                imagePointsList2f.remove(44);
                imagePointsList2f.remove(43);
                imagePointsList2f.remove(42);
                imagePointsList2f.remove(41);
                imagePointsList2f.remove(40);
//
                imagePointsList2f.remove(39);
//
                imagePointsList2f.remove(38);
                imagePointsList2f.remove(37);
                imagePointsList2f.remove(36);
                imagePointsList2f.remove(35);
                imagePointsList2f.remove(34);
                imagePointsList2f.remove(33);
//
                imagePointsList2f.remove(32);

                photoImagePoints2f.fromList(imagePointsList2f);
                photosImagePoints.add(photoImagePoints2f);

                Log.d("Debug:", "" + photoImagePoints2f.size() + fileDir.listFiles()[i].getPath());
            }
            Mat cameraMatrix = new Mat();
            MatOfDouble distCoeffs = new MatOfDouble();
            List<Mat> rvecs = new ArrayList<>();
            List<Mat> tvecs = new ArrayList<>();

            calibrateCamera(photosObjectPoints, photosImagePoints, IMAGE_SIZE, cameraMatrix, distCoeffs, rvecs, tvecs);

            Bitmap bitmap = BitmapFactory.decodeFile(fileDir.listFiles()[23].getPath());
            Mat matDistorted = new Mat();
            Mat matUndistorted = new Mat();
            Utils.bitmapToMat(bitmap, matDistorted);
            double h = matDistorted.size().height;
            double w = matDistorted.size().width;

            Mat newCameraMat = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, new org.opencv.core.Size(w, h), 1);

            undistort(matDistorted, matUndistorted, cameraMatrix, distCoeffs, newCameraMat);

            double mean_error = 0;
            MatOfPoint2f imgpoints2 = new MatOfPoint2f();
            for(int i = 0; i < photosObjectPoints.size(); i++) {
                projectPoints((MatOfPoint3f) photosObjectPoints.get(i), rvecs.get(i), tvecs.get(i), cameraMatrix, distCoeffs, imgpoints2);
                double error = norm(photosImagePoints.get(i), imgpoints2, NORM_L2) / imgpoints2.toList().size();
                mean_error += error;
            }
            putText(matUndistorted, "reprojection error: " + mean_error, new Point(100, 100),
                    FONT_HERSHEY_SIMPLEX, 1, COLOR_BLUE, 3);

            Utils.matToBitmap(matUndistorted, bitmap);
            File file = new File(getFilesDir(), "Calibrated.jpg");
            try {
                FileOutputStream out = new FileOutputStream(file);
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
                out.flush();
                out.close();
            } catch (Exception e) {

            }
        } else {

            for (int y = 0; y < 6; ++y) {
                for (int x = 0; x < 8; ++x) {
                    Point3 row = new Point3(x, y, 0);
                    pointsList.add(row);
                }
            }
//            pointsList.remove(47);
//            pointsList.remove(46);
//            pointsList.remove(45);
//            pointsList.remove(44);
//            pointsList.remove(43);
//            pointsList.remove(42);
//            pointsList.remove(41);
//            pointsList.remove(40);
//
//            pointsList.remove(39);
//
//            pointsList.remove(38);
//            pointsList.remove(37);
//            pointsList.remove(36);
//            pointsList.remove(35);
//            pointsList.remove(34);
//            pointsList.remove(33);
//
//            pointsList.remove(32);

            objectPoints.fromList(pointsList);

            File fileDir = getFilesDir();
            List<Mat> photosObjectPoints = new ArrayList<>();
            List<Mat> photosImagePoints = new ArrayList<>();
            MatOfPoint2f photoImagePoints2f = new MatOfPoint2f();
            List<Point> imagePoints2f;

            Bitmap bitmapRef = BitmapFactory.decodeFile(fileDir.listFiles()[0].getPath());
            Utils.bitmapToMat(bitmapRef, matTest);

            for (int i = 0; i < fileDir.listFiles().length; i++) {
                photosObjectPoints.add(objectPoints);
                imagePoints2f = new ArrayList<>(getChessboardCorners(BitmapFactory.decodeFile(fileDir.listFiles()[i].getPath())).toList());
                for (Point point : imagePoints2f) {
                    drawMarker(matTest, point, COLOR_RED, 1, 1, 2, 1);
                }
//                imagePoints2f.remove(47);
//                imagePoints2f.remove(46);
//                imagePoints2f.remove(45);
//                imagePoints2f.remove(44);
//                imagePoints2f.remove(43);
//                imagePoints2f.remove(42);
//                imagePoints2f.remove(41);
//                imagePoints2f.remove(40);
//
//                imagePoints2f.remove(39);
//
//                imagePoints2f.remove(38);
//                imagePoints2f.remove(37);
//                imagePoints2f.remove(36);
//                imagePoints2f.remove(35);
//                imagePoints2f.remove(34);
//                imagePoints2f.remove(33);
//
//                imagePoints2f.remove(32);

                photoImagePoints2f.fromList(imagePoints2f);
            /*for(int y=0; y<54; y++) {
                    Mat row = new Mat(1, 3, CV_32F);
                    row.col(0).setTo(new Scalar(photoImagePoints2f.toList().get(y).x));
                    row.col(1).setTo(new Scalar(photoImagePoints2f.toList().get(y).y));
                    row.col(2).setTo(new Scalar(0));
                    photoImagePoints.push_back(row);
            }*/
                Log.d("Debug:", "" + photoImagePoints2f.size() + fileDir.listFiles()[i].getPath());
                photosImagePoints.add(photoImagePoints2f);
            }
            Utils.matToBitmap(matTest, bitmapRef);
            File file2 = new File(getFilesDir(), "Calibrated2.jpg");
            try {
                FileOutputStream out = new FileOutputStream(file2);
                bitmapRef.compress(Bitmap.CompressFormat.PNG, 100, out);
                out.flush();
                out.close();
            } catch (Exception e) {
            }
            Mat cameraMatrix = new Mat();
            MatOfDouble distCoeffs = new MatOfDouble();
            List<Mat> rvecs = new ArrayList<>();
            List<Mat> tvecs = new ArrayList<>();

            calibrateCamera(photosObjectPoints, photosImagePoints, IMAGE_SIZE, cameraMatrix, distCoeffs, rvecs, tvecs);

            Bitmap bitmap = BitmapFactory.decodeFile(fileDir.listFiles()[0].getPath());
            Mat matDistorted = new Mat();
            Mat matUndistorted = new Mat();
            Utils.bitmapToMat(bitmap, matDistorted);
            double h = matDistorted.size().height;
            double w = matDistorted.size().width;

            Mat newCameraMat = getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, new org.opencv.core.Size(w, h), 1);

            undistort(matDistorted, matUndistorted, cameraMatrix, distCoeffs, newCameraMat);

            double mean_error = 0;
            MatOfPoint2f imgpoints2 = new MatOfPoint2f();
            for(int i = 0; i < photosObjectPoints.size(); i++) {
                projectPoints((MatOfPoint3f) photosObjectPoints.get(i), rvecs.get(i), tvecs.get(i), cameraMatrix, distCoeffs, imgpoints2);
                double error = norm(photosImagePoints.get(i), imgpoints2, NORM_L2) / imgpoints2.toList().size();
                mean_error += error;
            }
            putText(matUndistorted, "reprojection error: " + mean_error, new Point(100, 100),
                    FONT_HERSHEY_SIMPLEX, 1, COLOR_BLUE, 3);

            Utils.matToBitmap(matUndistorted, bitmap);
            File file = new File(getFilesDir(), "Calibrated.jpg");
            try {
                FileOutputStream out = new FileOutputStream(file);
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
                out.flush();
                out.close();
            } catch (Exception e) {
            }

            return;
        }
    }

    private MatOfPoint2f getAssymetricCircleCenters(Bitmap bitmap) {
        Mat matColor = new Mat();
        Mat matGrey = new Mat();

        Utils.bitmapToMat(bitmap, matColor);
        cvtColor(matColor, matGrey, COLOR_BGR2GRAY);

        threshold(matGrey, matGrey,127,255, THRESH_BINARY);
//        normalize(matGrey, matGrey, 0, 255, NORM_MINMAX);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        findContours(matGrey, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        double maxArea = 0.0;
        MatOfPoint bestContour = new MatOfPoint();
        for(MatOfPoint contour : contours) {
            double area = contourArea(contour);
            if (area > 1000.0) {
                if (area > maxArea) {
                    maxArea = area;
                    bestContour = contour;
                }
            }
        }

        List<MatOfPoint> bestContourList = new ArrayList<>();
        bestContourList.add(bestContour);

        Mat maskMatrix = zeros(new org.opencv.core.Size(matGrey.width(), matGrey.height()), CV_8U);

        drawContours(maskMatrix, bestContourList, 0, new Scalar(255.0, 255.0, 255.0, 255.0), -1);
        drawContours(maskMatrix, bestContourList, 0, new Scalar(0.0, 0.0, 0.0, 0.0), 2);

        Core.bitwise_and(matGrey, maskMatrix, matGrey);

        maskMatrix = zeros(new org.opencv.core.Size(matGrey.width() + 2, matGrey.height() + 2), CV_8U);
        floodFill(matGrey, maskMatrix, new Point(0,0), new Scalar(255, 255));

        SimpleBlobDetector_Params blobParams = new SimpleBlobDetector_Params();

        blobParams.set_minThreshold(8);
        blobParams.set_maxThreshold(255);
        blobParams.set_filterByCircularity(false);
        blobParams.set_filterByInertia(false);
//        blobParams.set_minInertiaRatio((float) 0.01);

        SimpleBlobDetector blobDetector = SimpleBlobDetector.create(blobParams);

        MatOfKeyPoint matOfKeyPoint = new MatOfKeyPoint();
        blobDetector.detect(matGrey, matOfKeyPoint);

        List<KeyPoint> orderedPoints = matOfKeyPoint.toList();

        MatOfPoint2f bestContour2f = new MatOfPoint2f(bestContour.toArray());
        RotatedRect rotatedRectangle = minAreaRect(bestContour2f);

        double angle = rotatedRectangle.angle;

        Point bl = new Point(9999, 0);
        Point tl = new Point(0, 9999);
        Point br = new Point(0, -9999);
        Point tr = new Point(-9999, 0);

        for (KeyPoint p : orderedPoints) {
            if (p.pt.x < bl.x) {
                bl = p.pt;
            }
            if (p.pt.x > tr.x) {
                tr = p.pt;
            }
            if (p.pt.y < tl.y) {
                tl = p.pt;
            }
            if (p.pt.y > br.y) {
                br = p.pt;
            }
        }
//        for(Point corner : orderedPoints) {
//            drawMarker(matColor, corner, COLOR_RED, 1, 2, 2, 1);
//        }

        if (angle > 45) {
            Point temp = new Point(tl.x, tl.y);
            tl = tr;
            tr = temp;
            temp = new Point(bl.x,bl.y);
            bl = tl;
            tl = temp;
            temp = new Point(bl.x,bl.y);
            bl = br;
            br = temp;
        }

//        putText(matColor, "tl", tl, FONT_HERSHEY_SIMPLEX, 1, COLOR_RED);
//        putText(matColor, "tr", tr, FONT_HERSHEY_SIMPLEX, 1, COLOR_RED);
//        putText(matColor, "bl", bl, FONT_HERSHEY_SIMPLEX, 1, COLOR_RED);
//        putText(matColor, "br", br, FONT_HERSHEY_SIMPLEX, 1, COLOR_RED);

//        Mat chessboardRotationMatrix = getRotationMatrix2D(rotatedRectangle.center, angle, 1);

        putText(matColor, "points = " + orderedPoints.size(), new Point(200,200), FONT_HERSHEY_SIMPLEX, 1, COLOR_GREEN);

//        Collections.sort(orderedPoints, new Comparator<KeyPoint>() {
//            public int compare(KeyPoint x1, KeyPoint x2) {
//                double x1Prime = chessboardRotationMatrix.get(0,0)[0] * x1.pt.x + chessboardRotationMatrix.get(0,1)[0] * x1.pt.y + chessboardRotationMatrix.get(0,2)[0];
//                double y1Prime = chessboardRotationMatrix.get(1,0)[0] * x1.pt.x + chessboardRotationMatrix.get(1,1)[0] * x1.pt.y + chessboardRotationMatrix.get(1,2)[0];
//                double x2Prime = chessboardRotationMatrix.get(0,0)[0] * x2.pt.x + chessboardRotationMatrix.get(0,1)[0] * x2.pt.y + chessboardRotationMatrix.get(0,2)[0];
//                double y2Prime = chessboardRotationMatrix.get(1,0)[0] * x2.pt.x + chessboardRotationMatrix.get(1,1)[0] * x2.pt.y + chessboardRotationMatrix.get(1,2)[0];
//                return Double.compare(100 * y1Prime + 10 * x1Prime, 100 * y2Prime + 10 * x2Prime);
//            }
//        });
//
//        Mat cornerMatrix = zeros(circleGridSize, CV_64FC2);
//
//        double smallestDeltaX = 9999;
//        double smallestDeltaY = 9999;
//
//        if(orderedPoints.size() <= circleGridSize.width * circleGridSize.height && orderedPoints.size() > circleGridSize.width * circleGridSize.height / 4) {
//            CAN_TAKE_PHOTO = true;
//
//            int currentRow = 0;
//            int currentColumn = 0;
//            double currentY = chessboardRotationMatrix.get(1, 0)[0] * orderedPoints.get(0).pt.x + chessboardRotationMatrix.get(1, 1)[0] * orderedPoints.get(0).pt.y + chessboardRotationMatrix.get(1, 2)[0];
//            double currentX = chessboardRotationMatrix.get(0, 0)[0] * orderedPoints.get(0).pt.x + chessboardRotationMatrix.get(0, 1)[0] * orderedPoints.get(0).pt.y + chessboardRotationMatrix.get(0, 2)[0];
//            double realX = orderedPoints.get(0).pt.x;
//            double realY = orderedPoints.get(0).pt.y;
//            for (KeyPoint corner : orderedPoints) {
//                double yPrime = chessboardRotationMatrix.get(1, 0)[0] * corner.pt.x + chessboardRotationMatrix.get(1, 1)[0] * corner.pt.y + chessboardRotationMatrix.get(1, 2)[0];
//                double XPrime = chessboardRotationMatrix.get(0, 0)[0] * corner.pt.x + chessboardRotationMatrix.get(0, 1)[0] * corner.pt.y + chessboardRotationMatrix.get(0, 2)[0];
//                if(abs(yPrime - currentY) > 15) {
//                    if (abs(corner.pt.y - realY) < smallestDeltaY) {
//                        smallestDeltaY = abs(corner.pt.y - realY);
//                    }
//                    currentRow++;
//                    currentColumn = 0;
//                } else if (abs(XPrime - currentX) > 15 && abs(corner.pt.x - realX) < smallestDeltaX) {
//                    smallestDeltaX = abs(corner.pt.x - realX);
//                }
//                cornerMatrix.put(currentRow, currentColumn, corner.pt.x, corner.pt.y);
//                currentColumn++;
//                currentY = yPrime;
//            }
//        } else {
//            CAN_TAKE_PHOTO = false;
//        }

        Point center = new Point();
        for (KeyPoint point : orderedPoints) {
            center.x += point.pt.x;
            center.y += point.pt.y;
        }
        center.x /= orderedPoints.size();
        center.y /= orderedPoints.size();

        double height1 = sqrt(pow(tl.x - bl.x, 2) + pow(tl.y - bl.y, 2));
        double height2 = sqrt(pow(tr.x - br.x, 2) + pow(tr.y - br.y, 2));
        double height = max(height1, height2);

        double width1 = sqrt(pow(tl.x - tr.x, 2) + pow(tl.y - tr.y, 2));
        double width2 = sqrt(pow(bl.x - br.x, 2) + pow(bl.y - br.y, 2));
        double width = max(width1, width2);

        MatOfPoint2f matrix1 = new MatOfPoint2f(tl, tr, bl, br);
        MatOfPoint2f matrix2;
        if (angle < 45) {
            matrix2 = new MatOfPoint2f(new Point(0, 0), new Point(sqrt(pow(width, 2) - pow(height / 6, 2)), height / 6), new Point(0, height),
                    new Point(sqrt(pow(width, 2) - pow(height / 6, 2)), height * 7 / 6));
        } else {
            matrix2 = new MatOfPoint2f(new Point(0, 0), new Point(width, 0), new Point(height / 6, height * 7 / 6),
                    new Point(11 / 10 * width, height * 7 / 6));
        }

        Mat warpMat = getPerspectiveTransform(matrix2, matrix1);

        Collections.sort(orderedPoints, new Comparator<KeyPoint>() {
            public int compare(KeyPoint x1, KeyPoint x2) {
                List<Point> listForMat = new ArrayList<>();
                listForMat.add(x1.pt);
                listForMat.add(x2.pt);
                MatOfPoint2f mat = new MatOfPoint2f();
                mat.fromList(listForMat);
                perspectiveTransform(mat, mat, warpMat.inv());
                if (abs(mat.toArray()[0].y - mat.toArray()[1].y) < 10) {
                    return Double.compare(mat.toArray()[0].x, mat.toArray()[1].x);
                }
                return Double.compare(mat.toArray()[0].y, mat.toArray()[1].y);
            }
        });

//        for (int i = 0; i < 1; i++) {
//            Point predictionPoint = new Point(cornerMatrix.get(i, 0)[0] + smallestDeltaX, cornerMatrix.get(i, 0)[1]);
//            int skipped = 0;
//            for (int j = 0; j < cornerMatrix.cols(); j++) {
//                if (j == 0) {
//                    cornerMatrixDoneWell.put(i, j, cornerMatrix.get(i, j));
//                    putText(matColor, String.valueOf(j), new Point(cornerMatrix.get(i, j)), FONT_HERSHEY_SIMPLEX, 1, COLOR_RED);
//                } else if (abs(cornerMatrix.get(i, j - skipped)[0] - predictionPoint.x) >= smallestDeltaX * 1.33 || abs(cornerMatrix.get(i, j - skipped)[1] - predictionPoint.y) >= smallestDeltaY * 1.33) {
//                    predictionPoint = new Point(predictionPoint.x + smallestDeltaX, predictionPoint.y);
//                    cornerMatrixDoneWell.put(i, j, 0, 0);
//                    skipped++;
//                    Log.d("Debug", "sdsd");
//                } else {
//                    cornerMatrixDoneWell.put(i, j, cornerMatrix.get(i, j - skipped));
//                    predictionPoint = new Point(cornerMatrix.get(i, j - skipped)[0] + smallestDeltaX, cornerMatrix.get(i, j - skipped)[1]);
//                    putText(matColor, String.valueOf(j), new Point(cornerMatrix.get(i, j - skipped)), FONT_HERSHEY_SIMPLEX, 1, COLOR_RED);
//                }
//            }
//        }

        List<Point> finalPoints = new ArrayList<>();

        for (KeyPoint keyPoint : orderedPoints) {
//            drawMarker(matColor, keyPoint.pt, COLOR_RED, 1, 2, 2, 1);
//            putText(matColor, String.valueOf(orderedPoints.indexOf(keyPoint)), keyPoint.pt, FONT_HERSHEY_SIMPLEX, 1, COLOR_RED);
            finalPoints.add(keyPoint.pt);
        }

        MatOfPoint2f finalMat = new MatOfPoint2f();
        finalMat.fromList(finalPoints);

        return finalMat;
    }

    private MatOfPoint2f getChessboardCorners(Bitmap bitmap) {

        MatOfPoint2f finalMat = new MatOfPoint2f();

        Mat matColor = new Mat();
        Mat matGrey = new Mat();
        Utils.bitmapToMat(bitmap, matColor);
        cvtColor(matColor, matGrey, COLOR_BGR2GRAY);
        threshold(matGrey, matGrey,127,255, THRESH_BINARY);
//        normalize(matGrey, matGrey, 0, 255, NORM_MINMAX);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        findContours(matGrey, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        double maxArea = 0.0;
        MatOfPoint bestContour = new MatOfPoint();
        for(MatOfPoint contour : contours) {
            double area = contourArea(contour);
            if (area > 1000.0) {
                if (area > maxArea) {
                    maxArea = area;
                    bestContour = contour;
                }
            }
        }

        List<MatOfPoint> bestContourList = new ArrayList<>();
        bestContourList.add(bestContour);

        Mat maskMatrix = zeros(new org.opencv.core.Size(matGrey.width(), matGrey.height()), CV_8U);

        if (bestContourList.size() > 0 && bestContourList.get(0) != null) {
            drawContours(maskMatrix, bestContourList, 0, new Scalar(255.0, 255.0, 255.0, 255.0), -1);
            drawContours(maskMatrix, bestContourList, 0, new Scalar(0.0, 0.0, 0.0, 0.0), 2);
            Core.bitwise_and(matGrey, maskMatrix, matGrey);
        }

        maskMatrix = zeros(new org.opencv.core.Size(matGrey.width() + 2, matGrey.height() + 2), CV_8U);

        floodFill(matGrey, maskMatrix, new Point(0,0), new Scalar(255, 255));

        MatOfPoint corners = new MatOfPoint();

        Mat emptyMat = new Mat();

//        Canny(matGrey, matGrey, 30, 150, 3, false);

        goodFeaturesToTrack(matGrey, corners, (int) (chessboardSize.width * chessboardSize.height), 0.01, 10, emptyMat, 3, false, 0.04);
//        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new org.opencv.core.Size(2, 2));
//        Imgproc.dilate(matGrey, matGrey, kernel);

        class Pair<L,R> {
            private L l;
            private R r;
            public Pair(L l, R r){
                this.l = l;
                this.r = r;
            }
            public L getL(){ return l; }
            public R getR(){ return r; }
            public void setL(L l){ this.l = l; }
            public void setR(R r){ this.r = r; }
        }

        MatOfPoint2f bestContour2f = new MatOfPoint2f(bestContour.toArray());
        RotatedRect rotatedRectangle = minAreaRect(bestContour2f);

        double angle = rotatedRectangle.angle;

        putText(matColor, "angle = " + angle, new Point(100, 200), FONT_HERSHEY_SIMPLEX, 1, COLOR_RED);

        Mat linesMat = new Mat();
        List<Pair<Double, Double>> lines = new ArrayList<>();
//        List<Pair<Double, Double>> extremeLines = new ArrayList<>();

//        HoughLines(matGrey, linesMat,1,CV_PI / 180,100);
//
//        boolean isRedundant;
//        for(int i = 0; i < linesMat.rows(); i++) {
//            double rho = linesMat.get(i, 0)[0];
//            double theta = linesMat.get(i, 0)[1];
//            isRedundant = false;
//            for (Pair l : lines) {
////                Log.d("Debug", "" + theta * 180 / CV_PI);
//                if (abs((double) l.getL() - rho) < 15 && (abs((double) l.getR() * 180 / CV_PI - theta * 180 / CV_PI) < 20
//                        || abs((double) l.getR() * 180 / CV_PI - (180 - theta * 180 / CV_PI)) < 20)) {
//                    isRedundant = true;
//                    break;
//                }
//            }
//            if (!isRedundant && (theta * 180 / CV_PI - angle) > 30) {
//                lines.add(new Pair<>(rho, theta));
//            }
////            if (!isRedundant && (abs(angle - theta * 180 / CV_PI) < 5 || abs(angle + 90 - theta * 180 / CV_PI) < 5
////            ||  abs(angle + 180 - theta * 180 / CV_PI) < 3)) {
////                lines.add(new Pair<>(rho, theta));
////            }
//        }
//        double lowestY = 9999;
//        double highestY =-9999;
//        double lowestX = 9999;
//        double highestX =-9999;
//        List<Point> lowestLine = new ArrayList<>();
//        List<Point> highestLine = new ArrayList<>();
//        List<Point> leftLine = new ArrayList<>();
//        List<Point> rightLine = new ArrayList<>();
//
//        lowestLine.add(0, new Point());
//        lowestLine.add(1, new Point());
//        highestLine.add(0, new Point());
//        highestLine.add(1, new Point());
//        leftLine.add(0, new Point());
//        leftLine.add(1, new Point());
//        rightLine.add(0, new Point());
//        rightLine.add(1, new Point());
//
//        for (Pair p : lines) {
//            double rho = (double) p.getL();
//            double theta = (double) p.getR();
//            double cosTheta = cos(theta);
//            double sinTheta = sin(theta);
//            double x0 = cosTheta * rho;
//            double y0 = sinTheta * rho;
//            Point P1 = new Point(x0 + 10000 *  (-sinTheta), y0 + 10000 * cosTheta);
//            Point P2 = new Point(x0 - 10000 *  (-sinTheta), y0 - 10000 * cosTheta);
//            if (y0 < lowestY) {
//                highestLine.set(0, P1);
//                highestLine.set(1, P2);
//                lowestY = y0;
//            }
//            if (y0 > highestY) {
//                lowestLine.set(0, P1);
//                lowestLine.set(1, P2);
//                highestY = y0;
//            }
//            if (x0 < lowestX) {
//                leftLine.set(0, P1);
//                leftLine.set(1, P2);
//                lowestX = x0;
//            }
//            if (x0 > highestX) {
//                rightLine.set(0, P1);
//                rightLine.set(1, P2);
//                highestX = x0;
//            }
//            line(matColor, P1, P2, COLOR_RED, 2);
//        }
//
        List<Point> orderedPoints = new ArrayList<>();
////
//        if (lowestLine.size() > 1 && highestLine.size() > 1
//        && leftLine.size() > 1 && rightLine.size() > 1) {
//            line(matColor, leftLine.get(0), leftLine.get(1), COLOR_RED, 2);
//        }

//        List<Point> intersectionPoints = new ArrayList<>();
//        if (lines.size() > 1) {
//            for (int i = 0; i < lines.size(); i++) {
//                double rho1 = lines.get(i).getL();
//                double theta1 = lines.get(i).getR();
//                double cost1 = cos(theta1);
//                double sint1 = sin(theta1);
//                for (int j = i + 1; j < lines.size(); j++) {
//                    double rho2 = lines.get(j).getL();
//                    double theta2 = lines.get(j).getR();
//                    double cost2 = cos(theta2);
//                    double sint2 = sin(theta2);
//                    double d = cost1 * sint2 - sint1 * cost2;
//                    if (abs(theta1 - theta2) > 0.083 * CV_PI) { // 15 degrees
//                        double xIntersection = (sint2 * rho1 - sint1 * rho2) / d;
//                        double yIntersection = (-cost2 * rho1 + cost1 * rho2) / d;
//                        intersectionPoints.add(new Point(xIntersection, yIntersection));
//                    }
//                }
//            }
//        }

//        for (Point corner : corners.toArray()) {
//            for (Point gridPoint : intersectionPoints) {
//                if (abs(corner.x - gridPoint.x) < 30 && abs(corner.y - gridPoint.y) < 30) {
//                    orderedPoints.add(corner);
//                    break;
//                }
//            }
//        }

        orderedPoints.addAll(corners.toList());

        Point bl = new Point(9999, 0);
        Point tl = new Point(0, 9999);
        Point br = new Point(0, -9999);
        Point tr = new Point(-9999, 0);

        for (Point p : orderedPoints) {
            if (p.x < bl.x) {
                bl = p;
            }
            if (p.x > tr.x) {
                tr = p;
            }
            if (p.y < tl.y) {
                tl = p;
            }
            if (p.y > br.y) {
                br = p;
            }
        }
//        for(Point corner : orderedPoints) {
//            drawMarker(matColor, corner, COLOR_RED, 1, 2, 2, 1);
//        }

        if (angle > 45) {
            Point temp = new Point(tl.x, tl.y);
            tl = tr;
            tr = temp;
            temp = new Point(bl.x,bl.y);
            bl = tl;
            tl = temp;
            temp = new Point(bl.x,bl.y);
            bl = br;
            br = temp;
        }

//        putText(matColor, "tl", tl, FONT_HERSHEY_SIMPLEX, 1, COLOR_RED);
//        putText(matColor, "tr", tr, FONT_HERSHEY_SIMPLEX, 1, COLOR_RED);
//        putText(matColor, "bl", bl, FONT_HERSHEY_SIMPLEX, 1, COLOR_RED);
//        putText(matColor, "br", br, FONT_HERSHEY_SIMPLEX, 1, COLOR_RED);


//        Mat boxCorners = new Mat();
//
//        boxPoints(rotatedRectangle, boxCorners);

//        circle(matColor, new Point(boxCorners.get(0,0)[0], boxCorners.get(0,1)[0]), 4, COLOR_RED, -1);
//        circle(matColor, new Point(boxCorners.get(1,0)[0], boxCorners.get(1,1)[0]), 4, COLOR_RED, -1);
//        circle(matColor, new Point(boxCorners.get(2,0)[0], boxCorners.get(2,1)[0]), 4, COLOR_RED, -1);
//        circle(matColor, new Point(boxCorners.get(3,0)[0], boxCorners.get(3,1)[0]), 4, COLOR_RED, -1);

        Point center = new Point();
        for (Point point : orderedPoints) {
            center.x += point.x;
            center.y += point.y;
        }
        center.x /= orderedPoints.size();
        center.y /= orderedPoints.size();

//        List<Point> extremePoints = new ArrayList<>();
//
        if (orderedPoints.size() > 3) {

//            circle(matColor, center, 2, COLOR_RED);
//
//            List<Point> top = new ArrayList<>();
//            List<Point> bottom = new ArrayList<>();
//
//            for (Point point : extremePoints) {
//                if (point.y < center.y) {
//                    top.add(point);
//                } else {
//                    bottom.add(point);
//                }
//            }
//
//            Point tl;
//            Point tr;
//            Point bl;
//            Point br;

//            if (top.size() > 1 && bottom.size() > 1) {
//                if (top.get(0).x < top.get(1).x) {
//                    tl = top.get(0);
//                    tr = top.get(1);
//                } else {
//                    tl = top.get(1);
//                    tr = top.get(0);
//                }
//
//                if (bottom.get(0).x < bottom.get(1).x) {
//                    bl = bottom.get(0);
//                    br = bottom.get(1);
//                } else {
//                    bl = bottom.get(1);
//                    br = bottom.get(0);
//                }

            double height1 = sqrt(pow(tl.x - bl.x, 2) + pow(tl.y - bl.y, 2));
            double height2 = sqrt(pow(tr.x - br.x, 2) + pow(tr.y - br.y, 2));
            double height = max(height1, height2);

            double width1 = sqrt(pow(tl.x - tr.x, 2) + pow(tl.y - tr.y, 2));
            double width2 = sqrt(pow(bl.x - br.x, 2) + pow(bl.y - br.y, 2));
            double width = max(width1, width2);

            MatOfPoint2f matrix1 = new MatOfPoint2f(tl, tr, bl, br);
            MatOfPoint2f matrix2 = new MatOfPoint2f(new Point(0, 0), new Point(width - 1, 0), new Point(0, height - 1), new Point(width - 1, height - 1));

//            putText(matColor, "tl1", new Point(0, 0), FONT_HERSHEY_SIMPLEX, 1, COLOR_GREEN);
//            putText(matColor, "tr1", new Point(width - 1, 0), FONT_HERSHEY_SIMPLEX, 1, COLOR_GREEN);
//            putText(matColor, "bl1", new Point(0, height - 1), FONT_HERSHEY_SIMPLEX, 1, COLOR_GREEN);
//            putText(matColor, "br1", new Point(width - 1, height - 1), FONT_HERSHEY_SIMPLEX, 1, COLOR_GREEN);

            Mat warpMat = getPerspectiveTransform(matrix2, matrix1);

//            MatOfPoint2f unrotatedCorners = new MatOfPoint2f(corners.toArray());
//            perspectiveTransform(unrotatedCorners, unrotatedCorners, warpMat.inv());
//
//            for(Point p : unrotatedCorners.toArray()) {
//                circle(matColor, p, 4, COLOR_RED, -1);
//            }

            Collections.sort(orderedPoints, new Comparator<Point>() {
                public int compare(Point x1, Point x2) {
                    List<Point> listForMat = new ArrayList<>();
                    listForMat.add(x1);
                    listForMat.add(x2);
                    MatOfPoint2f mat = new MatOfPoint2f();
                    mat.fromList(listForMat);
                    perspectiveTransform(mat, mat, warpMat.inv());
                    if (abs(mat.toArray()[0].y - mat.toArray()[1].y) < 10) {
                        return Double.compare(mat.toArray()[0].x, mat.toArray()[1].x);
                    }
                    return Double.compare(mat.toArray()[0].y, mat.toArray()[1].y);
                }
            });

//            for (Point p : orderedPoints) {
//                putText(matColor, "" + orderedPoints.indexOf(p), p, FONT_HERSHEY_SIMPLEX, 1, COLOR_RED);
//            }


//            for (int i = 0; i < orderedPoints.size(); i++) {
//                finalMat.put((int) (i / chessboardSize.height), (int) (i % chessboardSize.width), orderedPoints.get(i).x, orderedPoints.get(i).y);
//            }

            finalMat.fromList(orderedPoints);
        }
//
////
//
////        for(int index = 0; index < corners.toList().size() - 1; index++) {
////            if(!(sqrt(pow(corners.toArray()[index].x - corners.toArray()[index + 1].x, 2) + pow(corners.toArray()[index].y - corners.toArray()[index + 1].y, 2)) < 15)) {
////                cornersReduced.add(corners.toArray()[index]);
////            }
////        }
//
//        putText(matColor, "angle = " + angle, new Point(100,100), FONT_HERSHEY_SIMPLEX, 1, COLOR_GREEN);
//        putText(matColor, "corners = " + orderedPoints.size(), new Point(200,200), FONT_HERSHEY_SIMPLEX, 1, COLOR_GREEN);
//        Mat chessboardRotationMatrix = getRotationMatrix2D(rotatedRectangle.center, angle, 1);
//
//        Collections.sort(orderedPoints, new Comparator<Point>() {
//            public int compare(Point x1, Point x2) {
//                double x1Prime = chessboardRotationMatrix.get(0,0)[0] * x1.x + chessboardRotationMatrix.get(0,1)[0] * x1.y + chessboardRotationMatrix.get(0,2)[0];
//                double y1Prime = chessboardRotationMatrix.get(1,0)[0] * x1.x + chessboardRotationMatrix.get(1,1)[0] * x1.y + chessboardRotationMatrix.get(1,2)[0];
//                double x2Prime = chessboardRotationMatrix.get(0,0)[0] * x2.x + chessboardRotationMatrix.get(0,1)[0] * x2.y + chessboardRotationMatrix.get(0,2)[0];
//                double y2Prime = chessboardRotationMatrix.get(1,0)[0] * x2.x + chessboardRotationMatrix.get(1,1)[0] * x2.y + chessboardRotationMatrix.get(1,2)[0];
//                if (abs(y2Prime - y1Prime) < 15) {
//                    return Double.compare(x1Prime, x2Prime);
//                }
//                return Double.compare(y1Prime, y2Prime);
//            }
//        });

//        Mat cornerMatrix = zeros(chessboardSize, CV_64FC2);
//        if(orderedPoints.size() <= chessboardSize.width * chessboardSize.height
//        && orderedPoints.size() > 3) {
//            int currentRow = 0;
//            int currentColumn = 0;
//            double currentY = chessboardRotationMatrix.get(1, 0)[0] * orderedPoints.get(0).x + chessboardRotationMatrix.get(1, 1)[0] * orderedPoints.get(0).y + chessboardRotationMatrix.get(1, 2)[0];;
//            for (Point corner : orderedPoints) {
//                double yPrime = chessboardRotationMatrix.get(1, 0)[0] * corner.x + chessboardRotationMatrix.get(1, 1)[0] * corner.y + chessboardRotationMatrix.get(1, 2)[0];
//                if(abs(yPrime - currentY) > 20) {
//                    currentRow++;
//                    currentColumn = 0;
//                }
//                cornerMatrix.put(currentRow, currentColumn, corner.x, corner.y);
//                currentColumn++;
//                currentY = yPrime;
//            }
//        }

//        for(Point corner : orderedPoints) {
//            drawMarker(matColor, corner, COLOR_RED, 1, 2, 2, 1);
//            putText(matColor, String.valueOf(orderedPoints.indexOf(corner)), corner, FONT_HERSHEY_SIMPLEX, 1, COLOR_RED);
//        }

        return finalMat;
    }

    private MatOfPoint2f detectCcTags(Bitmap bitmap) {

        Mat matColor = new Mat();
        Mat matGrey = new Mat();
        Mat matGreyAdapted = new Mat();
        Mat hierarchy = new Mat();
        Mat uselessHierarchy = new Mat();
        List<MatOfPoint> uselessContours = new ArrayList<>();

        Utils.bitmapToMat(bitmap, matColor);

        cvtColor(matColor, matGrey, COLOR_BGR2GRAY);

        threshold(matGrey, matGreyAdapted, 127, 255, THRESH_BINARY);
        //normalize(matGreyAdapted, matGreyAdapted, 0, 255, NORM_MINMAX);

        findContours(matGreyAdapted, uselessContours, uselessHierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        double maxArea = 0.0;
        MatOfPoint bestContour = new MatOfPoint();
        for (MatOfPoint contour : uselessContours) {
            double area = contourArea(contour);
            if (area > 1000.0) {
                if (area > maxArea) {
                    maxArea = area;
                    bestContour = contour;
                }
            }
        }

        List<MatOfPoint> bestContourList = new ArrayList<>();
        bestContourList.add(bestContour);

        Mat maskMatrix = zeros(new org.opencv.core.Size(matGrey.width(), matGrey.height()), CV_8U);
        if (bestContour.toList().size() > 0) {
            drawContours(maskMatrix, bestContourList, -1, new Scalar(255.0, 255.0, 255.0, 255.0), -1);
            drawContours(maskMatrix, bestContourList, -1, new Scalar(0.0, 0.0, 0.0, 0.0), 2);
        }
        Core.bitwise_and(matGreyAdapted, maskMatrix, matGreyAdapted);

        maskMatrix = zeros(new org.opencv.core.Size(matGreyAdapted.width() + 2, matGreyAdapted.height() + 2), CV_8U);
        floodFill(matGreyAdapted, maskMatrix, new Point(0, 0), new Scalar(255, 255));

        List<RotatedRect> ellipses = new ArrayList<>();
        List<MatOfPoint> contoursInEllipseOrder = new ArrayList<>();
        List<MatOfPoint> contoursNew = new ArrayList<>();
        findContours(matGreyAdapted, contoursNew, hierarchy, RETR_LIST, CHAIN_APPROX_NONE);
        if (contoursNew.size() >= 6) {
            MatOfPoint2f tempMat2f = new MatOfPoint2f();
            for (int i = contoursNew.size() - 1; i >= 0; i--) {
                if (contoursNew.get(i).size().height > 5000) {
                    contoursNew.remove(i);
                    continue;
                }
                if (contoursNew.get(i).toArray().length > 4) {
                    contoursNew.get(i).convertTo(tempMat2f, CV_32F);
                    RotatedRect minAreaRect = minAreaRect(tempMat2f);
                    ellipses.add(minAreaRect);
                    contoursInEllipseOrder.add(contoursNew.get(i));
//                    ellipse(matColor, minAreaRect, COLOR_RED, 2);
                }
            }
        }

        List<RotatedRect> ellipsesWithCommonCentersBest = new ArrayList<>();
        List<RotatedRect> ellipsesWithCommonCenters = new ArrayList<>();
        List<MatOfPoint> contoursTemp = new ArrayList<>();
        List<MatOfPoint> contoursInBestEllipseOrder = new ArrayList<>();

        for (int i = 0; i < ellipses.size(); i++) {
            ellipsesWithCommonCenters.add(ellipses.get(i));
            contoursTemp.add(contoursInEllipseOrder.get(i));
            for (int j = i + 1; j < ellipses.size(); j++) {
                if ((sqrt(pow(ellipses.get(i).center.x - ellipses.get(j).center.x, 2) + pow(ellipses.get(i).center.y - ellipses.get(j).center.y, 2))) < 50) {
                    ellipsesWithCommonCenters.add(ellipses.get(j));
                    contoursTemp.add(contoursInEllipseOrder.get(j));
                }
            }
            if (ellipsesWithCommonCenters.size() > ellipsesWithCommonCentersBest.size()) {
                ellipsesWithCommonCentersBest.clear();
                ellipsesWithCommonCentersBest.addAll(ellipsesWithCommonCenters);
                contoursInBestEllipseOrder.clear();
                contoursInBestEllipseOrder.addAll(contoursTemp);
            }
            ellipsesWithCommonCenters.clear();
            contoursTemp.clear();
        }
//
//        for (int i = 0; i < ellipsesWithCommonCentersBest.size(); i++) {
//            ellipse(matColor, ellipsesWithCommonCentersBest.get(i), COLOR_RED, 2);
//        }
//
        Map<Integer, Point> imagePointsMap = new HashMap<>();
        List<Point> imagePoints = new ArrayList<>();
        if (ellipsesWithCommonCentersBest.size() > 1) {
            Point averageCenter = new Point(0,0);

            MatOfPoint biggestEllipseContour = contoursInBestEllipseOrder.get(0);
            RotatedRect biggestEllipse = ellipsesWithCommonCentersBest.get(0);

            Moments moments;
            Point center = new Point(0,0);

            for (int i = 0; i < ellipsesWithCommonCentersBest.size(); i++) {
                moments = moments(contoursInBestEllipseOrder.get(i));
                center.x = moments.m10 / moments.m00;
                center.y = moments.m01 / moments.m00;
                averageCenter.x += center.x;
                averageCenter.y += center.y;
                if (ellipsesWithCommonCentersBest.get(i).size.height > biggestEllipse.size.height
                        && ellipsesWithCommonCentersBest.get(i).size.width > biggestEllipse.size.width) {
                    biggestEllipse = ellipsesWithCommonCentersBest.get(i);
                    biggestEllipseContour = contoursInBestEllipseOrder.get(i);
                }
            }
            averageCenter.x /= ellipsesWithCommonCentersBest.size();
            averageCenter.y /= ellipsesWithCommonCentersBest.size();

            circle(matColor, averageCenter, 5, COLOR_RED, -1);

            List<Point> mostOuterEllipsePoints = new ArrayList<>();
            List<Point> mostOuterEllipsePoints2 = new ArrayList<>();
            for (int i = 0; i < 360; i+=36) {
                double s = sin(i*CV_PI/180);
                double c = cos(i*CV_PI/180);
                int iterationsToSkip = 0;
                Point P2 = new Point(averageCenter.x+s*150, averageCenter.y+c*150);
                double dy = P2.y - averageCenter.y;
                double dx = P2.x - averageCenter.x;
                double m = dy / dx;
                for (Point ringPoint : biggestEllipseContour.toArray()) {
                    if (iterationsToSkip != 0) {
                        iterationsToSkip--;
                        continue;
                    }
                    if (dx > 50 && abs(m * (ringPoint.x - averageCenter.x) - (ringPoint.y - averageCenter.y)) < 2) {
                        iterationsToSkip = 30;
//                        putText(matColor, "" + i, ringPoint, FONT_HERSHEY_SIMPLEX, 0.8, COLOR_RED, 1);
                        mostOuterEllipsePoints2.add(ringPoint);
//                        circle(matColor, ringPoint, 5, COLOR_RED, -1);
                    }
                }
            }
            for (int i = 0; i < mostOuterEllipsePoints2.size(); i += 2) {
                mostOuterEllipsePoints.add(mostOuterEllipsePoints2.get(i));
            }
            for (int i = 1; i < mostOuterEllipsePoints2.size(); i += 2) {
                mostOuterEllipsePoints.add(mostOuterEllipsePoints2.get(i));
            }
//            for (int i = 0; i < 10; i++) {
//                if (i == 0 || i == 5) {
//                    continue;
//                }
//                Point ringPoint = biggestEllipseContour.toArray()[i * (biggestEllipseContour.toArray().length / 10)];
//                putText(matColor, "" + i, ringPoint, FONT_HERSHEY_SIMPLEX, 0.8, COLOR_RED, 1);
//                mostOuterEllipsePoints.add(ringPoint);
//            }
//

            contoursNew.removeAll(contoursInBestEllipseOrder);
            List<Point> allPoints = new ArrayList<>();
            MatOfPoint mergedMat = new MatOfPoint();
            List<Integer> mergedIndexes = new ArrayList<>();
            List<MatOfPoint> finalContours = new ArrayList<>();

            finalContours.addAll(contoursInBestEllipseOrder);

            for (int i = 0; i < contoursNew.size(); i++) {
                allPoints.clear();
                allPoints.addAll(contoursNew.get(i).toList());
                double length1 = sqrt(pow(contoursNew.get(i).toArray()[0].x - averageCenter.x, 2) + pow(contoursNew.get(i).toArray()[0].y - averageCenter.y, 2));
                for (int j = i + 1; j < contoursNew.size(); j++) {
                    if (!mergedIndexes.contains(j)) {
                        double length2 = sqrt(pow(contoursNew.get(j).toArray()[0].x - averageCenter.x, 2) + pow(contoursNew.get(j).toArray()[0].y - averageCenter.y, 2));
                        if (abs(length1 - length2) < 100) {
                            allPoints.addAll(contoursNew.get(j).toList());
                            mergedIndexes.add(j);
                        }
                    }
                }
                if (!mergedIndexes.contains(i)) {
                    mergedMat.fromList(allPoints);
                    finalContours.add(mergedMat);
                }
            }
//
            Point finalAverageCenter = averageCenter;
            Collections.sort(finalContours, new Comparator<MatOfPoint>() {
                public int compare(MatOfPoint m1, MatOfPoint m2) {
                    double length1 = sqrt(pow(m1.toArray()[0].x - finalAverageCenter.x, 2) + pow(m1.toArray()[0].y - finalAverageCenter.y, 2));
                    double length2 = sqrt(pow(m2.toArray()[0].x - finalAverageCenter.x, 2) + pow(m2.toArray()[0].y - finalAverageCenter.y, 2));
                    return Double.compare(length1, length2);
                }
            });
//
            Map<Integer, Point> currentRingPointsMap = new HashMap<>();
            boolean isThisMergedContour;
            Integer currentId = 0;
            Integer previouslyAddedId = 0;
            for (int i = 0; i < finalContours.size(); i++) {
                isThisMergedContour = false;
                for (int j = 0; j < mostOuterEllipsePoints.size(); j++) {
                    double dy = mostOuterEllipsePoints.get(j).y - averageCenter.y;
                    double dx = mostOuterEllipsePoints.get(j).x - averageCenter.x;
                    double m = dy / dx;
                    int iterationsToSkip = 0;
                    currentRingPointsMap.clear();
                    if (isThisMergedContour) {
                        currentId = j * 4 + i * 6;
                    } else {
                        currentId = j * 2 + i * 8;
                    }
                    for (Point ringPoint : finalContours.get(i).toArray()) {
                        if (iterationsToSkip != 0) {
                            iterationsToSkip--;
                            continue;
                        }
                        if (dx > 50 && abs(m * (ringPoint.x - averageCenter.x) - (ringPoint.y - averageCenter.y)) < 2) {
                            iterationsToSkip = 30;
//                            if (!isThisMergedContour && currentRingPointsMap.size() > 0 &&
//                                    sqrt(pow(ringPoint.x - currentRingPointsMap.get(previouslyAddedId).x, 2) +
//                                    pow(ringPoint.y - currentRingPointsMap.get(previouslyAddedId).y, 2)) < 100) {
//                                Point prevPoint = currentRingPointsMap.get(previouslyAddedId);
//                                currentRingPointsMap.remove(previouslyAddedId);
//                                if (sqrt(pow(ringPoint.x - mostOuterEllipsePoints.get(j).x, 2) +
//                                        pow(ringPoint.y - mostOuterEllipsePoints.get(j).y, 2)) > 200) {
//                                    currentId += j * 2 + 2;
//                                    currentRingPointsMap.put(previouslyAddedId + j * 2 + 2, prevPoint);
//                                } else {
//                                    currentId += j * 2;
//                                    currentRingPointsMap.put(previouslyAddedId + j * 2, prevPoint);
//                                }
//                                Log.d("Debug", "merge??");
//                                isThisMergedContour = true;
//                            }
                            currentRingPointsMap.put(currentId, ringPoint);
//                            circle(matColor, ringPoint, 5, COLOR_RED, -1);
                            previouslyAddedId = currentId;
                            currentId++;
                        }
                    }
                    int finalJ = j;
                    List<Point> points = new ArrayList<>(currentRingPointsMap.values());
                    Collections.sort(points, new Comparator<Point>() {
                        public int compare(Point p1, Point p2) {
                            double length1 = sqrt(pow(p1.x - mostOuterEllipsePoints.get(finalJ).x, 2) +
                                    pow(p1.y - mostOuterEllipsePoints.get(finalJ).y, 2));
                            double length2 = sqrt(pow(p2.x - mostOuterEllipsePoints.get(finalJ).x, 2) +
                                    pow(p2.y - mostOuterEllipsePoints.get(finalJ).y, 2));
                            return Double.compare(length1, length2);
                        }
                    });
                    for (int k = 0; k < points.size(); k++) {
//                        if (isThisMergedContour) {
////                            switch (k % 2) {
////                                case 0 :
////                                    imagePointsMap.put((Integer) currentRingPointsMap.keySet().toArray()[k] - j * 2 - k / 2, points.get(k));
////                                    break;
////                                case 1 :
////                                    imagePointsMap.put((Integer) currentRingPointsMap.keySet().toArray()[k] + 5 - j * 2 - k / 2, points.get(k));
////                                    break;
////                            }
//                        } else {
                            imagePointsMap.put((Integer) currentRingPointsMap.keySet().toArray()[k], points.get(k));
//                        }
                    }
                }
            }
//
            MatOfPoint2f bestContour2f = new MatOfPoint2f(bestContour.toArray());
            RotatedRect rotatedRectangle = minAreaRect(bestContour2f);

            double angle = rotatedRectangle.angle;
            if (angle > 45) {
                angle -= 90;
            }

//            Mat ccTagRotationMatrix = getRotationMatrix2D(averageCenter, -angle, 1);

            Map<Integer, Point> unrotatedImagePointsMap = new HashMap<>();

//            for (Integer i : imagePointsMap.keySet()) {
//                double xPrime = ccTagRotationMatrix.get(0,0)[0] * imagePointsMap.get(i).x + ccTagRotationMatrix.get(0,1)[0] * imagePointsMap.get(i).y
//                        + ccTagRotationMatrix.get(0,2)[0];
//                double yPrime = ccTagRotationMatrix.get(1,0)[0] * imagePointsMap.get(i).x + ccTagRotationMatrix.get(1,1)[0] * imagePointsMap.get(i).y
//                        + ccTagRotationMatrix.get(1,2)[0];
//                unrotatedImagePointsMap.put(i, new Point(xPrime, yPrime));
//            }

//            putText(matColor, "angle = " + angle, new Point(200,200), FONT_HERSHEY_SIMPLEX, 1, COLOR_GREEN);
//
            for (Integer i : imagePointsMap.keySet()) {
                imagePoints.add(imagePointsMap.get(i));

                putText(matColor, "" + i, imagePointsMap.get(i), FONT_HERSHEY_SIMPLEX, 0.8, COLOR_RED, 1);
            }
        }

//        if (contoursNew.toArray().length > 6) {
//            List<Point> allPoints = new ArrayList<>();
//            allPoints.addAll(contoursNew.get(1).toList());
//            allPoints.addAll(contoursNew.get(6).toList());
//            MatOfPoint mat = new MatOfPoint();
//            mat.fromList(allPoints);
//            mat.convertTo(tempMat2f, CV_32F);
//            RotatedRect rectangle = minAreaRect(tempMat2f);
//            ellipse(matColor, rectangle, COLOR_RED, 3);
//        }

//        int depth;
//        Double contourId;
//        Double outerRingId = -1.0;
//
//        RotatedRect elli = new RotatedRect();
//
//        MatOfPoint2f contour2f = new MatOfPoint2f();
//        org.opencv.core.Point meanCenter;
////        int orderId = 0;
//
//        List<Point> outerRingOfPoints = new ArrayList<>();
//        List<Point> imagePoints = new ArrayList<>();
//        List<Point> imagePointsReduced = new ArrayList<>();
//
//        List<Double> contourIds = new ArrayList<>();
//
//        for (int i = 0; i < contours.size(); i++) {
//
//            List<Double> h = new ArrayList<>();
//
//            for(double d : hierarchy.get(0, i)) {
//                h.add(d);
//            }
//
//            depth = 0;
//
//            if (h.get(2) == -1 && h.get(3) != -1) {
//
//                depth++;
//                contourId = h.get(3);
//                contours.get(i).convertTo(contour2f, CV_32F);
//
//                if(contour2f.toList().size() > 4) {
//
//                    elli = fitEllipse(contour2f);
//                    meanCenter = elli.center;
//
//                    contourIds.add(hierarchy.get(0, contourId.intValue())[2]);
//
////                    ellipse(matColor, elli, COLOR_RED, 4);
//
//                    while(depth != 5 && hierarchy.get(0, contourId.intValue())[3] != -1) {
//
//                        contourIds.add(contourId);
//
//                        depth++;
//                        contourId = hierarchy.get(0, contourId.intValue())[3];
//                        contours.get(contourId.intValue()).convertTo(contour2f, CV_32F);
//
//                        if(hierarchy.get(0, contourId.intValue())[3] == -1) {
//                            contourIds.add(contourId);
//                        }
//
//                        if(contour2f.toList().size() > 4) {
//
//                            elli = fitEllipse(contour2f);
//
//                            Point[] contour2fArray = contour2f.toArray();
//                            //Log.d("Debug", String.valueOf(elli.angle));
//                            if(depth == 5) {
//                                outerRingId = contourId;
////                                ellipse(matColor, elli, COLOR_RED, 4);
//                                for (int j = 0; j < 8; j++) {
//                                    Point ringPoint = contour2fArray[j * (contour2f.toList().size() / 8)];
//                                    //PointF ringPointF = new PointF((float) ringPoint.x, (float) ringPoint.y);
////                                    circle(matColor, ringPoint, 3, COLOR_RED, -1);
//                                    outerRingOfPoints.add(ringPoint);
//                                }
//                            }
//
//                            meanCenter.x = (meanCenter.x + elli.center.x) / 2;
//                            meanCenter.y = (meanCenter.y + elli.center.y) / 2;
//                        }
//                    }
//                    //matReference = matColor;
//
//                    if(depth == 5) {
//                        contours.get(outerRingId.intValue()).convertTo(contour2f, CV_32F);
//                        //Log.d("Debug:", String.valueOf(contour2f.size()));
//                        elli = fitEllipse(contour2f);
//
////                        ellipse(matColor, elli, COLOR_RED, 4);
//
//                        meanCenter.x = (meanCenter.x + elli.center.x) / 2;
//                        meanCenter.y = (meanCenter.y + elli.center.y) / 2;
//
//                        MatOfPoint2f contourInner2f = new MatOfPoint2f();
//
//                        for (int j = 0; j < 8; j++) {
//                            Point ringPoint = contour2f.toArray()[j * (contour2f.toList().size() / 8)];
//                            double dy = ringPoint.y - meanCenter.y;
//                            double dx = ringPoint.x - meanCenter.x;
//                            double m = dy / dx;
//                            int additionalPointsAdded = 0;
//                            for(Point point : contour2f.toArray()) {
//                                if(abs(dx) < 100 && (dy / abs(dy) == ((point.y - meanCenter.y) / abs(point.y - meanCenter.y))) && abs(point.x - meanCenter.x) < 1) {
//                                    //circle(matColor, point, 3, COLOR_RED, -1);
//                                    imagePoints.add(point);
//                                }
//                                if(dx > 100 && abs(m * (point.x - meanCenter.x) - (point.y - meanCenter.y)) < 1) {
//                                    //circle(matColor, point, 3, COLOR_RED, -1);
//                                    imagePoints.add(point);
//                                }
//                            }
//                            for(Double id : contourIds) {
//                                contours.get(id.intValue()).convertTo(contourInner2f, CV_32F);
//                                for(Point point : contourInner2f.toArray()) {
//                                    if(abs(dx) < 100 && (dy / abs(dy) == ((point.y - meanCenter.y) / abs(point.y - meanCenter.y))) && abs(point.x - meanCenter.x) < 1) {
//                                        //circle(matColor, point, 3, COLOR_RED, -1);
//                                        imagePoints.add(point);
//                                    }
//                                    if(dx > 100 && abs(m * (point.x - meanCenter.x) - (point.y - meanCenter.y)) < 1) {
//                                        //circle(matColor, point, 3, COLOR_RED, -1);
//                                        imagePoints.add(point);
//                                    } /*else if((point.x - meanCenter.x) - abs(dx) < 10) {
//                                        imagePoints.add(point);
//                                    }*/
//                                }
//                            }
//
//                        }
////                        ellipse(matColor, elli, COLOR_RED, 4);
////
////                        circle(matColor, meanCenter, 3, COLOR_RED, -1);
////                        putText(matColor, String.valueOf(orderId), new org.opencv.core.Point((int) meanCenter.x + 1, (int) meanCenter.y + 1),
////                                FONT_HERSHEY_SIMPLEX, 2, COLOR_BLUE, 3);
//
//                        for(int index = 0; index < imagePoints.size() - 1; index++) {
//                            if(!(sqrt(pow(imagePoints.get(index).x - imagePoints.get(index + 1).x, 2) + pow(imagePoints.get(index).y - imagePoints.get(index + 1).y, 2)) < 15)) {
//                                imagePointsReduced.add(imagePoints.get(index));
//                            }
//                        }
////                        putText(matColor, String.valueOf(imagePointsReduced.size()), new org.opencv.core.Point( 100, 100),
////                                FONT_HERSHEY_SIMPLEX, 2, COLOR_BLUE, 3);
//                        NUMBER_OF_CCTAG_IMAGE_POINTS = imagePointsReduced.size();
//                        if(NUMBER_OF_CCTAG_IMAGE_POINTS == 47 && TAKE_PHOTO == 1) {
//                            File file = new File(getFilesDir(), "Ref" + System.currentTimeMillis() + ".jpg");
//                            TAKE_PHOTO = 0;
//                            try {
//                                FileOutputStream out = new FileOutputStream(file);
//                                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
//                                out.flush();
//                                out.close();
//                            } catch (Exception e) {
//                            }
//                        }
//                    }
//                }
//            }
//        }

        MatOfPoint2f finalMat = new MatOfPoint2f();
        finalMat.fromList(imagePoints);

        return finalMat;
    }

    private void showAcceptedRejectedButton(boolean acceptedRejected) {
        if (acceptedRejected) {
            CameraX.unbind(preview, imageAnalysis);
            llBottom.setVisibility(View.VISIBLE);
            btnCapture.hide();
            textureView.setVisibility(View.GONE);
        } else {
            btnCapture.show();
            llBottom.setVisibility(View.GONE);
            textureView.setVisibility(View.VISIBLE);
            textureView.post(new Runnable() {
                @Override
                public void run() {
                    startCamera();
                }
            });
        }
    }


    private void updateTransform() {
        Matrix mx = new Matrix();
        float w = textureView.getMeasuredWidth();
        float h = textureView.getMeasuredHeight();

        float cX = w / 2f;
        float cY = h / 2f;

        int rotationDgr;
        int rotation = (int) textureView.getRotation();

        Log.d("Debug", "" + rotation);

//        switch (rotation) {
//            case Surface.ROTATION_0:
//                rotationDgr = 0;
//                break;
//            case Surface.ROTATION_90:
//                rotationDgr = 90;
//                break;
//            case Surface.ROTATION_180:
//                rotationDgr = 180;
//                break;
//            case Surface.ROTATION_270:
//                rotationDgr = 270;
//                break;
//            default:
//                return;
//        }

//        mx.postRotate((float) rotationDgr, cX, cY);
//        textureView.setTransform(mx);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {

        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera();
            } else {
                Toast.makeText(this, "Permissions not granted by the user.", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }

    private boolean allPermissionsGranted() {

        for (String permission : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, permission) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }


    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {

        switch (item.getItemId()) {
            case R.id.clear:
                clearDirectory();
                return true;

            case R.id.calibrate:
                calibration();
                return true;

            case R.id.chessboard:
                currentImageProcessing = "CHESSBOARD";
                startCamera();
                return true;

            case R.id.ccTag:
                currentImageProcessing = "CCTAG";
                startCamera();
                return true;

            case R.id.assymetricCircleGrid:
                currentImageProcessing = "ASSYMETRIC_CIRCLES";
                startCamera();
                return true;
        }

        return super.onOptionsItemSelected(item);
    }

    @Override
    public void onClick(View v) {
        switch (v.getId()) {

            case R.id.btnReject:
                showAcceptedRejectedButton(false);
                break;

            case R.id.btnAccept:
                File file = new File(
                        getFilesDir(), "" + System.currentTimeMillis() + "_JDCameraX.jpg");

                imageCapture.takePicture(file, new ImageCapture.OnImageSavedListener() {
                    @Override
                    public void onImageSaved(@NonNull File file) {
                        showAcceptedRejectedButton(false);
                        try {
                            ExifInterface exif = new ExifInterface(file.getPath());
                            int orientation = exif.getAttributeInt(
                                    ExifInterface.TAG_ORIENTATION,
                                    ExifInterface.ORIENTATION_NORMAL);

                            int angle = 0;

                            if (orientation == ExifInterface.ORIENTATION_ROTATE_90) {
                                angle = 90;
                            } else if (orientation == ExifInterface.ORIENTATION_ROTATE_180) {
                                angle = 180;
                            } else if (orientation == ExifInterface.ORIENTATION_ROTATE_270) {
                                angle = 270;
                            }

                            Matrix mat = new Matrix();
                            mat.postRotate(angle);
                            BitmapFactory.Options options = new BitmapFactory.Options();
                            options.inSampleSize = 2;

                            Bitmap bmp = BitmapFactory.decodeStream(new FileInputStream(file),
                                    null, options);
                            Bitmap bitmap = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(),
                                    bmp.getHeight(), mat, true);
                            FileOutputStream fout = new FileOutputStream(file);
                            bitmap.compress(Bitmap.CompressFormat.PNG, 100,
                                    fout);

                        } catch (IOException e) {
                            Log.w("TAG", "-- Error in setting image");
                        } catch (OutOfMemoryError oom) {
                            Log.w("TAG", "-- OOM Error in setting image");
                        }
                        try {
                            Exif exif = Exif.createFromFile(file);

                        } catch(IOException e) {

                        }
                        Toast.makeText(getApplicationContext(), "Image saved successfully in Pictures Folder", Toast.LENGTH_LONG).show();
                    }

                    @Override
                    public void onError(@NonNull ImageCapture.UseCaseError useCaseError, @NonNull String message, @Nullable Throwable cause) {

                    }
                });
                break;
        }
    }
}
