import './index.pug'

import 'semantic-ui-offline/semantic.css'
import './app.sass'

import faceModule from './face.js'
import svmModule from './svm.js'

window.$ = window.jQuery = require('jquery')

const video = document.getElementsByTagName('video')[0]
const canvas = document.getElementsByTagName('canvas')[0]

const width = 320
const height = 240

let faceCollection = []
let faceId = null

let isCapture = false
let isValidate = false
let captureProcess = null

video.setAttribute('width', width)
video.setAttribute('height', height)
canvas.setAttribute('width', width)
canvas.setAttribute('height', height)

navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia

Promise.all([svmModule.init(), faceModule.init()]).then(main).catch(err => console.log(err))

function snapshot() {
  const context = canvas.getContext('2d')
  context.drawImage(video, 0, 0, width, height)
  return context.getImageData(60, 20, 200, 200)
}

function main() {
  const successCallback = (stream) => {
    video.srcObject = stream
  }

  const errorCallback = (error) => {
    console.log("navigator.getUserMedia error: ", error)
    alert('Your browser doesn\'t support WebRTC API')
  }

  navigator.getUserMedia({audio: false, video: true}, successCallback, errorCallback)

  $('#capture').click(e => {

    $(e.currentTarget).toggleClass('teal').toggleClass('red')
    isCapture = !isCapture

    if (isCapture) {
      $(e.currentTarget).text('Stop')
      captureProcess = setInterval(() => faceCollection.push(snapshot()), 200)
    } else {
      clearInterval(captureProcess)

      $(e.currentTarget).text('Build Your Face ID ... ')

      setTimeout(() => {
        faceId = faceCollection.length >= 10 ? faceModule.transformMore(faceCollection) : faceModule.transform(faceCollection)
        $(e.currentTarget).text('Build One Class SVM ...')
        svmModule.buildOCSVM(faceId)
        $(e.currentTarget).text(faceId.length)
        faceCollection.length = 0
      }, 2000)

    }
  })

  $('#validate').click(e => {
    if (!faceId || faceId.length === 0) {
      alert('You don\'t have face id.')
    }

    if (isCapture) {
      return
    }

    $(e.currentTarget).toggleClass('blue').toggleClass('red')
    isValidate = !isValidate

    if (isValidate) {
      $(e.currentTarget).text('Stop validate').ready(() => {

        captureProcess = setInterval(() => {
          const target = faceModule.transform([snapshot()])[0]
          const prediction = svmModule.predict(target[0])

          console.log(prediction)

          if (prediction !== -1) {
            $('#validate').click()
            $('#lock').fadeOut('slow')
            $('#unlock').fadeIn('slow')
          }
        }, 500)
      })

    } else {
      clearInterval(captureProcess)
      $(e.currentTarget).text('Validate')
    }
  })

  $('#relock').click(e => {
    $('#unlock').fadeOut('slow')
    $('#lock').fadeIn('slow')
  })
}
