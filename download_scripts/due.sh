cd $DATASET_DIR

echo "Donwloading DocVQA dataset..."
mkdir docvqa
cd docvqa
wget https://applica-public.s3.eu-west-1.amazonaws.com/due/datasets/DocVQA.tar.gz
tar xvf DocVQA.tar.gz && rm DocVQA.tar.gz
cd ..

echo "Donwloading InfoVQA dataset..."
mkdir infovqa
cd infovqa
wget https://applica-public.s3.eu-west-1.amazonaws.com/due/datasets/InfographicsVQA.tar.gz
tar xvf InfographicsVQA.tar.gz && rm InfographicsVQA.tar.gz
cd ..

echo "Donwloading TabFact dataset..."
mkdir tabfact
cd tabfact
wget https://applica-public.s3.eu-west-1.amazonaws.com/due/datasets/TabFact.tar.gz
tar xvf TabFact.tar.gz && rm TabFact.tar.gz
cd ..

echo "Donwloading WTQ dataset..."
mkdir wtq
cd wtq
wget https://applica-public.s3.eu-west-1.amazonaws.com/due/datasets/WikiTableQuestions.tar.gz
tar xvf WikiTableQuestions.tar.gz && rm WikiTableQuestions.tar.gz
cd ..

echo "Donwloading KLC dataset..."
mkdir klc
cd klc
wget https://applica-public.s3.eu-west-1.amazonaws.com/due/datasets/KleisterCharity.tar.gz
tar xvf KleisterCharity.tar.gz && rm KleisterCharity.tar.gz
cd ..

echo "Donwloading DeepForm dataset..."
mkdir deepform
cd deepform
wget https://applica-public.s3.eu-west-1.amazonaws.com/due/datasets/DeepForm.tar.gz
tar xvf DeepForm.tar.gz && rm DeepForm.tar.gz
cd ..

echo "Donwloading PWC dataset..."
mkdir pwc
cd pwc
wget https://applica-public.s3.eu-west-1.amazonaws.com/due/datasets/PWC.tar.gz
tar xvf PWC.tar.gz && rm PWC.tar.gz
