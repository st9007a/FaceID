import './index.pug'

import * as tf from '@tensorflow/tfjs'
import { loadFrozenModel } from '@tensorflow/tfjs-converter'

const MODEL_URL = './model/tensorflowjs_model.pb'
const WEIGHTS_URL = './model/weights_manifest.json'

const model = loadFrozenModel(MODEL_URL, WEIGHTS_URL).then(() => console.log('done!'))
/*
document.getElementsByTagName('h1')[0].textContent = '1';
if (!('navigator' in window)) {
  document.getElementsByTagName('h1')[0].textContent = 'Not support';
} else {
  document.getElementsByTagName('h1')[0].textContent = 'support';
}

navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia;

var constraints = {audio: false, video: true};

function successCallback(stream) {
  window.stream = stream;
  document.getElementsByTagName('h1')[0].textContent = 'success';
}

function errorCallback(error){
  console.log("navigator.getUserMedia error: ", error);
  document.getElementsByTagName('h1')[0].textContent = 'fail';
}

navigator.getUserMedia(constraints, successCallback, errorCallback);
*/
