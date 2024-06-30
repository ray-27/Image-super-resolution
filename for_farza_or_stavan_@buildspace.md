- Hi here i am putting all the updates of my work on the AI model that i am building in the Season 5 nights & weekends.
- I am creating this readme file so that it become easy for you to see my progress, and helps to land me an [intership at buildspace](https://buildspace.so/intern)
 

- **All the context are in descending order.**
- [[#Week 2 (Building a toy version)]]
- [[#Week 1 (Finding the idea!)]]


I can be found on [sage](https://sage.buildspace.so/@rajveer-yadav-1xfHiyq) ,[linkedin](https://www.linkedin.com/in/rajveer-yadav-3264b722a/) , [X](https://x.com/ray_2_7) , [insta](https://www.instagram.com/__raydiation__?igsh=aHduMWdsanJmZWQ%3D&utm_source=qr) , [github](https://github.com/ray-27)

---
# Week 2 (Building a toy version)
- So in this week i prepared the dataset for the edge image creation dataset. you can see the code for the same in `edge_dataset.py` file
- Further more i have created a model to enhance the resolution of this image and increase the texture of the edges.
- I have taken inspiration from a model SRGAN, it is a _Generative Adversarial Network_ (_GAN_) that has shown good results in the field of super resolution. you can have a look at this model in the `model.py` file.
- I am going to train this model on Nvidia Graphic card and will compare the result as to how good this model works.
- __Further plans on the model:__ to try architectures like _transformers_ which are the building block of ChatGPT and Claude, they work better than the traditional CNN methods, so by next week i'll build a model on this architecture. 
- I'll tell the results in the submission on the n&w submission page.

# Week 1 (Finding the idea!)
- So i had this project in mind from a long time, i did a little research work in my third year of college in [Medal's lab](https://www.ee.iitb.ac.in/web/labs/medical-deep-learning-and-artificial-intelligence-lab-medal/) where i worked on the literature of image super resolution for medical images.
- Due to the limited resources it was hard for me to build a model from scratch. Now i am using cloud resources to get this done!
- So here's my one liner _an ai model that upscales image_. 
- My plan is to first start with building a model that upscales the resolution of image which has edges that is a grayscale image 

![[for_buildspace /edgemodel_idea.png|700]]

- then i'll work towards combining this model with the 3 channels RGB model on a similar but efficient architecture.