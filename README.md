# Student Attendance Project

### dataset structure : 

```
/project 
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
