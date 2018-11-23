# DNN Surgery
This project is for COMP5707 & COMP5708 Capstone Project in Unversity of Sydney

## Author
Yu Yang 

## Intruduction of Surgery
The project named DNN Surgery, which executes part of inference at the edge and the rest is processed at the strong backend. The major motivate is to accelerate the inference speed when the computational power is not so strong of the device. 

There are three series of experiment:

* First, five pre-trained models which are trained by Imagenet were compared from model size, load time and accuracy, the purpose of the experiment is to find the most suitable model which contains ring architecture to do the surgery. 
* The second series of experiment is for guide the final experiment, it gives some prerequisites of surgery. 
* The last experiment is for find the optimal wound and all desirable wounds which is also the final purpose of the project.


### Analogy with Organ Donation Surgery

DNN Surgery executes part of inference at the edge and the rest is processed at the strong backend.
The motivation is to accelerate the inference speed with the device which has no strong computation power. The surgery could be analogy with blood donation, there are five major parts in surgery: the frontend could be viewed as transplant who is people who have received blood donations, the backend could be seen as donar who helps the transplant, the intermediary of transfer in blood donation is an intravenous line which is like wireless Internet in DNN surgery. Wound is the position of cut layer, in blood donation, it is the position of blodd vessel. And incision could be compared as syrings. 

![alt text](https://github.com/yangyuchelsea/DNN-surgery/blob/master/surgery_experiment/result/surgery/analog.png)

### Architecture

Different with regular inference, the processing of inference need to be divided in two parts in DNN surgery. The wound is taken as the cut-off point, the frontend only operate from input to appointed intermediate layer(s) and send allocation instruction to backend and features of intermediate layer(s). The backend receives the instruction and features and executes from appointed intermediate layer(s) to the end of inference, then send the final output(classification) of DNN back to frontend. Both frontend and backend have the entire structure and weight of DNN.


### Support Module

* Graph model: plot model with index of layer in model
* Incision support module:
    * wound generation: generate candidate wound based on the architecture of model(include ring detector)
    * wound to incision: incision is more specific position than wound, it points out whether the surgery is before or after appointed intermediate layer(s)
    * feasible wound generation:candidate operation plan

 
## Software and Hardware Specification


* Frontend: raspberry pi 3

   * Model B \& 16GB NOOBS
   * SAMSUNG microSD 16GB


* Backend: Mac Pro

   * Processor 2.9 GHz Intel Core i5
   * Memory 8 GB


* Python version: python 3.6.6

   * Tensorflow: 1.11.0
* Keras: 2.2.2



## Surgery code
[Basic experiment](https://github.com/yangyuchelsea/DNN-surgery/blob/master/surgery_experiment/code/setup_exp.py)<br/>
[Find the optimal wounds](https://github.com/yangyuchelsea/DNN-surgery/blob/master/surgery_experiment/code/surgery_for_resnet50.py)

## Conclustion

From the experiments, we could get three major conclusion. First, DNN Surery is practicable of model with residual module and inception module in 4G network. Second, the optimal wound is after the $6^(th)$ layer/before $7^(th)$ layer, it could accelerate 65\% than the normal inference time when the first time to implement the surgery and accelerate 77\% after the first time. The size of intermediate transfer file is 803320 bytes, it needs 3ms to save the file and if the current network speed is faster than 7.453 Mbps, the surgery is effective and efficient. And the last, there are 33 desirable wounds, the surgery is effective as well in the wounds if the uplink and downlink network speed are both higher than 48 Mbps. The desirable wounds are the list:

|   1  |11, 14|40, 46|44, 48|
|------|:----:|:----:|:-----| 
|   2  |12, 14|40, 47|  49  |
|   3  |12, 16|41, 46|  50  |
|   4  |  17  |41, 48|  59  |
|   5  |  18  |42, 46|  60  | 
|   6  |  27  |42, 48|  70  |
| 7, 14|  28  |43, 46|      |
| 8, 14|39, 46|43, 48|      |
|11, 14|39, 48|44, 46|      |
            
   


## Visualization of result
[Visualization](https://github.com/yangyuchelsea/DNN-surgery/blob/master/surgery_experiment/result/surgery/visual.ipynb)<br/>
[Presentation Slides](https://github.com/yangyuchelsea/DNN-surgery/blob/master/Presentation.pdf)



