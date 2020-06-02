# MedAIR
## MedAIR Instructions


### #setup commands:

cd MedAIR # move to MedAIR directory

python setup.py develop --user #you're on your own for finding and installing dependencies

cd scripts

python build_db.py

python build_word_freq.py

python build_alignment_matrix.py # default embedding bio-wv, use --embedding to change

python build_elastic_search.py #requires elasticsearch to be running in background


### #Q&A commands:

cd MedAIR # move to MedAIR directory

cd alignment/retriever

python alignment_ranker.py # currently runs all embeddings, so make sure you perform build_alignment_matrix.py for all embeddings or change the code to only run the desired embedding


## Embedding Information


<table>
  <tr>
   <td>Embedding
   </td>
   <td>download link
   </td>
  </tr>
  <tr>
   <td>bio-wv
   </td>
   <td><a href="https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.vec.bin">https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.vec.bin</a> 
   </td>
  </tr>
  <tr>
   <td>glove
   </td>
   <td><a href="http://nlp.stanford.edu/data/glove.840B.300d.zip">http://nlp.stanford.edu/data/glove.840B.300d.zip</a> 
   </td>
  </tr>
  <tr>
   <td>pubmed-pmc-wv
   </td>
   <td><a href="http://evexdb.org/pmresources/vec-space-models/">http://evexdb.org/pmresources/vec-space-models/</a>
   </td>
  </tr>
  <tr>
   <td>wiki-pubmed-pmc-wv
   </td>
   <td><a href="http://evexdb.org/pmresources/vec-space-models/">http://evexdb.org/pmresources/vec-space-models/</a>
   </td>
  </tr>
</table>



<table>
  <tr>
   <td>Embedding
   </td>
   <td># docs
   </td>
   <td># question tokens
   </td>
   <td># embedded question tokens
   </td>
   <td>% question tokens embedded
   </td>
  </tr>
  <tr>
   <td>bio-wv
   </td>
   <td>231581
   </td>
   <td>32431
   </td>
   <td>28816
   </td>
   <td>88.9
   </td>
  </tr>
  <tr>
   <td>glove
   </td>
   <td>231581
   </td>
   <td>32431
   </td>
   <td>25124
   </td>
   <td>77.5
   </td>
  </tr>
  <tr>
   <td>pubmed-pmc-wv
   </td>
   <td>231581
   </td>
   <td>32431
   </td>
   <td>26459
   </td>
   <td>81.6
   </td>
  </tr>
  <tr>
   <td>wiki-pubmed-pmc-wv
   </td>
   <td>231581
   </td>
   <td>32431
   </td>
   <td>26745
   </td>
   <td>82.5
   </td>
  </tr>
</table>



## Alignment Results


<table>
  <tr>
   <td>es top 1000
<p>
top 30
   </td>
   <td>bio-wv
   </td>
   <td>pubmed-pmc-wv
   </td>
   <td>wiki-pubmed-pmc-wv
   </td>
   <td>glove
   </td>
  </tr>
  <tr>
   <td>dev
   </td>
   <td>25.4
   </td>
   <td>25.9
   </td>
   <td>27.2
   </td>
   <td>27.6
   </td>
  </tr>
  <tr>
   <td>test
   </td>
   <td>26.4
   </td>
   <td>26.6
   </td>
   <td>26.1
   </td>
   <td>25.7
   </td>
  </tr>
  <tr>
   <td>train
   </td>
   <td>25.7
   </td>
   <td>25.4
   </td>
   <td>25.1
   </td>
   <td>25.3
   </td>
  </tr>
</table>



<table>
  <tr>
   <td>es top 10_000
<p>
top 30
   </td>
   <td>bio-wv
   </td>
   <td>pubmed-pmc-wv
   </td>
   <td>wiki-pubmed-pmc-wv
   </td>
   <td>glove
   </td>
  </tr>
  <tr>
   <td>dev
   </td>
   <td>25.6
   </td>
   <td>25.5
   </td>
   <td>24.8
   </td>
   <td>28.3
   </td>
  </tr>
  <tr>
   <td>test
   </td>
   <td>26.8
   </td>
   <td>27.1
   </td>
   <td>28.3
   </td>
   <td>25.7
   </td>
  </tr>
  <tr>
   <td>train
   </td>
   <td>25.8
   </td>
   <td>25.8
   </td>
   <td>25.7
   </td>
   <td>25.1
   </td>
  </tr>
</table>



<table>
  <tr>
   <td>es top 1000
<p>
top 100
   </td>
   <td>bio-wv
   </td>
   <td>pubmed-pmc-wv
   </td>
   <td>wiki-pubmed-pmc-wv
   </td>
   <td>glove
   </td>
  </tr>
  <tr>
   <td>dev
   </td>
   <td>25.6
   </td>
   <td>26.2
   </td>
   <td>27.1
   </td>
   <td>28.5
   </td>
  </tr>
  <tr>
   <td>test
   </td>
   <td>25.7
   </td>
   <td>26.1
   </td>
   <td>25.6
   </td>
   <td>25.3
   </td>
  </tr>
  <tr>
   <td>train
   </td>
   <td>25.6
   </td>
   <td>25.3
   </td>
   <td>25.1
   </td>
   <td>25.2
   </td>
  </tr>
</table>



<table>
  <tr>
   <td>es top 1000
<p>
top 30
<p>
nltk tokenized
   </td>
   <td>bio-wv
   </td>
   <td>pubmed-pmc-wv
   </td>
   <td>wiki-pubmed-pmc-wv
   </td>
   <td>glove
   </td>
  </tr>
  <tr>
   <td>dev
   </td>
   <td>25.7
   </td>
   <td>27.2
   </td>
   <td>27.8
   </td>
   <td>26.5
   </td>
  </tr>
  <tr>
   <td>test
   </td>
   <td>26.5
   </td>
   <td>25.7
   </td>
   <td>26.2
   </td>
   <td>25.0
   </td>
  </tr>
  <tr>
   <td>train
   </td>
   <td>26.2
   </td>
   <td>25.6
   </td>
   <td>25.6
   </td>
   <td>25.7
   </td>
  </tr>
</table>



## DRQA TFIDF Results


<table>
  <tr>
   <td>topn
   </td>
   <td>100
   </td>
   <td>40
   </td>
   <td>30
   </td>
   <td>20
   </td>
   <td>10
   </td>
   <td>100
   </td>
   <td>30
   </td>
  </tr>
  <tr>
   <td>nltk tokenize
   </td>
   <td>False
   </td>
   <td>False
   </td>
   <td>False
   </td>
   <td>False
   </td>
   <td>False
   </td>
   <td>True
   </td>
   <td>True
   </td>
  </tr>
  <tr>
   <td>dev
   </td>
   <td>31.1
   </td>
   <td>31.5
   </td>
   <td>32.1
   </td>
   <td>32.2
   </td>
   <td>32.9
   </td>
   <td>30.4
   </td>
   <td>31.9
   </td>
  </tr>
  <tr>
   <td>test
   </td>
   <td>33.5
   </td>
   <td>32.6
   </td>
   <td>33.5
   </td>
   <td>32.7
   </td>
   <td>32.1
   </td>
   <td>32.3
   </td>
   <td>33.2
   </td>
  </tr>
  <tr>
   <td>train
   </td>
   <td>30.9
   </td>
   <td>31.8
   </td>
   <td>32.1
   </td>
   <td>31.5
   </td>
   <td>31.4
   </td>
   <td>30.3
   </td>
   <td>31.5
   </td>
  </tr>
</table>
