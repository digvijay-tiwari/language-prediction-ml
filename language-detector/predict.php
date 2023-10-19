<?php
require_once __DIR__ . '/../vendor/autoload.php';

use Phpml\FeatureExtraction\TfIdfTransformer;
use Phpml\FeatureExtraction\TokenCountVectorizer;
use Phpml\ModelManager;
use Phpml\Tokenization\WordTokenizer;

$strings = [
    "I will be there"
];

$filepath='svc.ini';
$modelManager = new ModelManager();
$restoredClassifier = $modelManager->restoreFromFile($filepath);

$s = file_get_contents('vectorizer');
$vectorizer = unserialize($s);

$s = file_get_contents('tfIdfTransformer');
$tfIdfTransformer = unserialize($s);

$vectorizer->transform($strings);

$tfIdfTransformer->transform($strings);

$result = $restoredClassifier->predict($strings);
print_r($result);