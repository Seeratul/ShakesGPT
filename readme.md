## GPT from Scratch
### Introduction 
This repository was build as part of my masters program in cognitive science for the intensive course "Building GPT from scratch".     
It is divided into 4 parts:   
- Task 1: In which basic data handling and Byte Pair encoding is covered     
- Task 2: In which I use those encodings with N-Grams to create a basic text completion feature    
- Task 3: In which I built a Neural encoding and back propagation from scratch   
- Task 4: In which I implement a General Purpose Transformer and then apply it to the dataset  

### Task 1

For code checkout [main1.py](main1.py).

All the measures for task 1 were eyeballed as proper HPO was supposed to wait till Task 2.    
 
To get a good first look at the data I ran:
```tr 'A-Z' 'a-z' < shakes.txt | tr -sc 'A-Za-z' '\n' | sort | uniq -c | sort -nr | head```
in the console.

After which I performed the following steps:
- Perform train and test split.
I used the provided file clean_nltk_shakespear_data.py to create a train test and eval set.
- Train the segmenter with varying k.
I used compression as a success metric, diminishing returns were reached at 2000-4000k with a compression of ~2.4 to ~2.5. 
I would also like to mention that I put a hard limit of 5 on the maximum segment length to help against overfitting onto long words.
- Compare the performance against a different set.
The compression was similar between the training data and test data with a difference of 0.03.
- Measure for accuracy.
I chose the delta between the test and the training data compression as my accuracy metric. 
I made the decision that massively increasing k for minimal returns (above 4000) was not worth it.

### Bonus: 
As BPE progressively merges subwords some subwords might end up "orphaned"/rarely used as a large chunk of their occurrence might get swallowed by a bigger subword.  
Given that we want as much information as possible packed into as few and as general subwords as possible this is not ideal.  
The solution to this was to run a second optimization/bonus round in which subwords that have less occurrence than the most common pair
get unmerged and that most common pair gets turned into a subword instead.  
The performance impact of this method will be discussed in Task 2.  
In task one with k = 2000 it improves compression on the training set by 0.002 with the test set being unaffected.  
This could be due to the small size of the dataset or because this post-processing is frankly a waste of time.  
Given minimal ks like 50 there is an improvement for the test data of 0.006 which is an improvement of compression by 5%.  
Note: For data symmetry reasons, the optimization/bonus round always leads to an optimal set after k-1 passes.  
I think this is really neat and probably something already known, somewhere.

### Task 2
For code checkout [main2.py](main2.py).

- N gram engine.
Arbitrarily scalable engine that uses the highest 3 grams that have a match for the key with a 0.6,0.3,0.1 weighting respectively. 
Automatically cuts text to size of the highest n-gram and can default to using only 2 (0.7,0.3) grams or the unigram model.   
Hardware limited on my computer to n<=10grams.
Applies Laplace smoothing on generation.
- [Ngramhpo.py](Ngramhpo.py).
Uses Ngram engine to calculate a matrix of perplexity relative to k/n for an engine generated with train.

![](/images/ngramtable.png)

- Extrinsic evaluation/sentencegen. 
Takes a tuple containing an arbitrary number of strings (1+), an n-gram model to use, the number of tokens to maximally generate, the top results to sample from (1 for deterministic), and utilizes end-of-sequence tokens (. ? !) for an early stop.    
Sadly, the generation is pure gibberish for the models with the lowest perplexity.
It would certainly be possible to provide nicer generation that is overfitted.


### Task 3
For code checkout [main3.py](main3.py).

Task 3 was programmed entirely without PyTorch with the exception of Legacy batching code.
The following components were hand made:
- A one hot encoder that turns the input into a one hot vector
- The embedding table
- The entire forward pass
- The entire backward pass including weight updates using sgd with batchsize 1 blocksize 1
- A version of the above that uses batches of larger sizes to speed up processing utilizing np's efficient matrix multiplications.
- Proper SGD 
- A generate function to utilize the model
- Saving and loading
- Calculating perplexity

I probably spent way too much time on this but I really enjoyed trying to work it all out. 
Highlights include problems with the learning rate, efficient matrix multiplication of 3d matrices and multiple hours
of (futilely) attempting to find out how I ended up having a '+' where there really should be a '-'.   
My implementation currently struggles with learning larger k's due to hardware limitations but performs adequately for k=1.    
While I would love to spend more time on this, I think
this is a respectable result given the time.       
In the demo with k=1 and a rather short runtime a perplexity of 26.13 
on the validation set was achieved.

![](/images/TLNeuralgramK1.png)

### Task 4
For code checkout [main4.py](main4.py).

Implementation:
Its a standard GPT implementation along the lines of the provided examples.
It contains:
- A self build transformer block using GELU
- An optimized data loader for batching
- The Gelu activation function
- A one call hook to run it for HPO with certain parameters
- A visualization tool
- A tool for ther generated output
- An early stopping hook with a setting for patience

For HPO I used my usual scaffolding with some slight tweaks available in [GPTHPO.py](GPTHPO.py).          
The tweaks include a seperate table that tracks for how many iterations each combination ran. I have not included them in the report for brevity's sake.  
As recommened I will limit my exploration of k to k = 2000 and one k = 1 run on the best parameters for 
k = 2000.   
The Parameters I chose to optimize were 
- Learning rate, as it limits how "fine" the result can be but also slows down the progress.    
- Depth as it is an efficent way to increase complexity.
- And lastly neural embedding size as it is costly but also necessary for performance.    
I decided to first explore Learning rate and the number of layers/depth.    

I got the following result:

![](/images/lr_d.png)

All of the runs ran untill early termination was reached.   
My scaling between min and max didn't really work as intended.   
Regardless, I got a pretty solid result that d ~ 8 seems to lead to the best results, with smaller d's not far behind.   
And lr should be 0.0025>0.00001.    
I will now run it again with d = 8 testing to find good parameters for lr and neural embedding size.  
![](/images/lr_ne.png)

All of the runs ran untill early termination was reached. 
Intrestingly the best run ne=72 lr=0.0005 took the longest. 
(As opposed to for example the runs with the lowerst lr which I had naively assumed would take longest.)
So those will be my final parameters.  
| Param | Val |
| -------- | ------- |
| Lr | 5e-4 | 
| ne | 72  |
| d | 8  |

Below the graph.  
Training loss in Blue.   
Test loss in Orange.  
And Perplexity in green on a wrong scale.

![](/images/Finalk2000.png)

Below the correct graph for Perplexity(Green)

![](/images/Finalk2000Perp.png)

It ran for 3200 iteration before being cut short as test loss stopped improving.  
Test loss ended up at 3.94.  
With test Perplexity being 34.72.  

For the test text 'Lord: Rise! My people, conquer the north!' it produced 
'urding, bufancie.' as output.  
Well at least the stopping generation on stop tokens works.

Now utilizing the same parameters but k=1
Training loss in Blue.   
Test loss in Orange.  

![](/images/tlk1final.png)

Perplexity in Blue Below

![](/images/Perpk1Final.png)

It ran for 13200 iteration before being cut short as test loss stopped improving.     
Test loss ended up at 1.67   
With test Perplexity being 5.34.    
For the test text 'Lord: Rise! My people, conquer the north!' it produced      
'i never is thee probo, peace!'     

And with those words from my best GPT implementation yet, I would like to finish this report.
In the end the lower complexity task of k=1 was easier to solve than having higher more discriptive k actually make sense.
I would be really curious if/how any student achiveved better or similar performance with higher k´s given the extremely limited size of the input data.
