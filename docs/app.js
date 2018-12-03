// load pre-trained model
let model;
tf.loadModel('./model/model.json')
    .then(pretrainedModel => {
        document.getElementById('inference-button').classList.remove('is-loading');
        pretrainedModel.summary();
        model = pretrainedModel;
    })
    .catch(error => {
        console.log(error);
    });

function getAccuracyScores(imageData) {
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

    // const imageData = getImageData();
    // const accuracyScores = getAccuracyScores(imageData);
    // const maxAccuracy = accuracyScores.indexOf(Math.max.apply(null, accuracyScores));
    // const elements = document.querySelectorAll(".accuracy");
    // elements.forEach(el => {
    //     el.parentNode.classList.remove('is-selected');
    //     const rowIndex = Number(el.dataset.rowIndex);
    //     if (maxAccuracy === rowIndex) {
    //         el.parentNode.classList.add('is-selected');
    //     }
    //     el.innerText = accuracyScores[rowIndex];
    // })
}
