// load pre-trained model
let model;
tf.loadModel('./model/model.json')
    .then(pretrainedModel => {
        document.getElementById('inference-button').classList.remove('is-loading');
        model = pretrainedModel;
    })
    .catch(error => {
        console.log(error);
    });

function getResponseSentence(imageData) {
    const score = tf.tidy(() => {
        // convert to tensor (shape: [width, height, channels])
        const channels = 1; // grayscale
        let input = tf.fromPixels(imageData, channels);
        // normalized
        input = tf.cast(input, 'float32').div(tf.scalar(255));
        // reshape input format (shape: [batch_size, width, height, channels])
        input = input.expandDims();
        // predict
        return model.predict(input).dataSync();
    });
    return score;
}

function inference() {
    console.log(document.getElementById("input_text").value);
    // getResponseSentence();
}
