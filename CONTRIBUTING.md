# Contributing    
Welcome to the PyTorch Lightning community! We're building the most advanced research platform on the planet to implement the latest, best practices that the amazing PyTorch team rolls out!  

## Lightning Design Principles   
We encourage all sorts of contributions you're interested in adding! When coding for lightning, please follow these principles.    

#### No PyTorch interference   
We don't want to add any abstractions on top of pure PyTorch. This gives researchers all the control they need without having to learn yet another framework.    

#### Simple Internal Code    
It's useful for users to look at the code and understand very quickly what's happening. Many users won't be engineers. Thus we need to value clear, simple code over condensed ninja moves. While that's super cool, this isn't the project for that :)      

#### Force User Decisions To Best Practices    
There are 1,000 ways to do something. However, something eventually becomes standard practice that everyone does. Thus we pick one way of doing it and force everyone to do it this way. A good example is accumulated gradients. There are many ways to implement, we just pick one and force users to use that one. A bad forced decision would be to make users use a specific library to do something.    

When something becomes a best practice, we add it to the framework. This likely looks like code in utils or in the model file that everyone keeps adding over and over again across projects. When this happens, bring that code inside the trainer and add a flag for it.

#### Simple External API    
What makes sense to you may not make sense to others. Create an issue with an API change suggestion and validate that it makes sense for others. Treat code changes how you treat a startup: validate that it's a needed feature, then add if it makes sense for many people.    

#### Gain User Trust    
As a researcher you can't have any part of your code going wrong. So, make thorough tests that ensure an implementation of a new trick or subbtle change is correct.    

## Contribution types    
Currently looking for help implementing new features or adding bug fixes. 

A lot of good work has already been done in project mechanics (requirements.txt, setup.py, pep8, badges, ci, etc...) we're in a good state there thanks to all the early contributors (even pre-beta release)!   

## Bug fixes:  
1. Submit a github issue.   
2. Fix it.  
3. Submit a PR! 

## New Features:  
1. Submit a github issue.   
2. We'll agree on the feature scope.     
3. Submit a PR! (with updated docs and tests 🙃).   
