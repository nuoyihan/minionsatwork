// -----JS CODE-----
//@input Component.Text text
//@input SceneObject ball

const categories = new Array("basketball", "bear", "skateboard");


var mlComponent = script.getSceneObject().getComponent("Component.MLComponent");
mlComponent.onLoadingFinished = onLoadingFinished;
mlComponent.inferenceMode = MachineLearning.InferenceMode.CPU;

var output = mlComponent.getOutput("logits");

function onLoadingFinished(){
    print("model built");
    //var scores_data = output.data();
    script.createEvent("UpdateEvent").bind(onUpdate);
}

function onUpdate(){
    var scores = output.data;
    var idx = argmax(scores);
    var probs = softmax(scores)
    var maxProb = maxval(probs);
    
    print("========Predicted:" + categories[idx] + "===========");
//    print("score for" + categories[0] + ":" + scores[0]);
//    print("score for" + categories[1] + ":" + scores[1]);
//    print("score for" + categories[2] + ":" + scores[2]);
    print("probability for " + categories[0] + " = " + probs[0]);
    print("probability for " + categories[1] + " = " + probs[1]);
    print("probability for " + categories[2] + " = " + probs[2]);
    
    if(maxProb>0.8){
        script.text.text = categories[idx];
        if (idx == 0){
            script.ball.enabled = true;
            script.text.text = categories[idx];
        }
    }
    else {
        script.text.text = "";
    }
}

//helper funtion
function argmax(arr){
    if (arr.length === 0) {
        return -1;
    }
    var max = arr[0];
    var maxIndex = 0;

    for (var i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIndex = i;
            max = arr[i];
        }
    }
    return maxIndex;
}

function softmax(logits){
    var probs = new Array(0.0, 0.0, 0.0)
    total = 0.0
    for (var i = 0 ; i < logits.length; i++) {
        total += Math.exp(logits[i])
    }
    for (var i = 0 ; i < logits.length; i++) {
        probs[i] = Math.exp(logits[i]) / total
    }
    return probs
}

function maxval(arr){
    if (arr.length === 0) {
        return -1;
    }
    var max = arr[0];
    var maxIndex = 0;

    for (var i = 1; i < arr.length; i++) {
        if (arr[i] > max) {
            maxIndex = i;
            max = arr[i];
        }
    }
    return max;
}

//var scores = mlComponent.getOutput("logits").data;
//print(scores);


//var mlComponent = script.testML;
//
//const categories = new Array("basketball", "bear", "skateboard");
////mlComponent.autoRun = true;
//print("test");
//
//var scores = mlComponent.getOutput("logits").data;
//print(scores);