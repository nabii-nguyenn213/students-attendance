# Student Attendance Project

### Dataset Preparation : 
- Prepare **at least 50 images** for each student's faces in different perspective. Save in folders which are named by their *Student_ID* in `student_attendance/dataset/Student_ID`.
- Run `Augmentation_dataset.py` to make more dataset of student's faces if needed (recommendation if dataset is limited). The `facenet_dataset` folder will automatically generate, which is used to extract feature of their faces using *facenet*. The feature embedding is passed in the `SVM` to train to classify students.

### Dataset structure : 

```
/students_attendance/ 
   |----dataset/
           |------students/
                       |------facenet_dataset/
                                |--------train/
                                          |------MSSV1/
                                                   |--------.jpg, .png, .jpeg     
                                          |------MSSV2/
                                                   |--------.jpg, .png, .jpeg    
                                          ...
                                          |------MSSV30/
                                                   |--------.jpg, .png, .jpeg   
                                |--------test/
                                          |------MSSV1/
                                                   |--------.jpg, .png, .jpeg     
                                          |------MSSV2/
                                                   |--------.jpg, .png, .jpeg    
                                          ...
                                          |------MSSV30/
                                                   |--------.jpg, .png, .jpeg  
                                |--------val/
                                          |------MSSV1/
                                                   |--------.jpg, .png, .jpeg     
                                          |------MSSV2/
                                                   |--------.jpg, .png, .jpeg    
                                          ...
                                          |------MSSV30/
                                                   |--------.jpg, .png, .jpeg  
                       |------MSSV1/
                                |--------.jpg, .png, .jpeg
                       |------MSSV2/
                                |--------.jpg, .png, .jpeg
                       |------MSSV3/
                                |--------.jpg, .png, .jpeg
                            ......
                       |------MSSV30/
                                |--------.jpg, .png, .jpeg
```

> `train/` , `test/` and `val/` folders will be automatically generated when running `Augmentation_dataset.py` file.

### Demo

<img src="demo.gif" width="100%">

### Face Classification Model

- I use `facenet` to extract feature of the student's faces, compress it into embeddings, which is used to train an `SVM` to classiy the student's faces.
- You can train a `FaceNet` model; however, in this project, just a few student's faces are trained, so a simple `SVM` is enough to achieve `>90% accuracy`.

### Anti-Spoofing Model

##### YOLOv8n

- The `YOLOv8` is added just for one reason: *phone detected*. It will detect if there are any phones in the frame. If there is any phone in the frame, the model will classify it a spoof. 
- The reason I added this into this project is to **speed up the model**, if YOLO detects any phones in the frame, the model will immediately mark that a spoof without running `anti-spoof-mn3`.
- In short, during attendance, no phone is allowed in the frame.

##### Anti-Spoof-mn3
- Due to many limited condition (hardware, dataset, ...), many model have been tested to suitable for this limitation. And I decide to pick [anti-spoof-mn3 model](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/anti-spoof-mn3) because of the following reasons :
    - **Pros** : 
        - `Anti-Spoof-mn3` is a small, light model and trained to predict whether or not a spoof RGB image given to the input. 
        - It's a best pre-trained model I tested so far. 
        - The model was built based on [MobileNetV3](https://arxiv.org/abs/1905.02244), trained on [CelebA-Spoof dataset](https://arxiv.org/abs/2007.12342). So that the model performs fast and high accuracy.

    - **Cons** : 
        - Because I've not fine-tuned the model (limited time and dataset), the model is very sensitive of light. 
        - Not as strong as large models (like EfficientNet, ResNet) in detecting very advanced attacks (hyper-realistic masks, digital manipulations).
        - Focuses on speed, so feature richness is limited compared to larger CNNs.

- **Summary** : 
    - This model is recommended if I use to run in real-time (even on CPU) because of its fast inference, lightweight.
    - However, the accuracy is not good if compare to the big model, and may struggle with new types of spoofing it wasn’t trained on.
    - This project used `anti-spoof-mn3` because of the following limitation : 
        - Time : This is final project in one of my subject at school. 
        - Hardware : If you have Depth Camera or 3D Camera, this model is not recommended. There are much more others well-accuracy models (e.g, 3D CNNs). 
        - Dataset : We are not able to collect dataset in a short time to fine-tuned this model. 
