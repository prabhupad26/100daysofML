//webkitURL is deprecated but nevertheless 
URL = window.URL || window.webkitURL;
var gumStream;
//stream from getUserMedia() 
var rec;
//Recorder.js object 
var input;
//MediaStreamAudioSourceNode we'll be recording 
// shim for AudioContext when it's not avb. 
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext = new AudioContext;
//new audio context to help us record 
var recordButton = document.getElementById("recordButton");
var stopButton = document.getElementById("stopButton");
var pauseButton = document.getElementById("pauseButton");
var voice_text = document.getElementById("computeButton");
var text_description = document.getElementById("text_description");
//add events to those 3 buttons
recordButton.addEventListener("click", startRecording);
pauseButton.addEventListener("click", pauseRecording);
voice_text.addEventListener("click", function(){
text_description.innerHTML = "Hello how ar you";})

function startRecording() { 
    console.log("recordButton clicked");
    recordButton.disabled = true;
    stopButton.disabled = false;
    pauseButton.disabled = false;


    const handleSuccess = function(stream) {
        const options = {mimeType: 'audio/webm'};
        const mediaRecorder = new MediaRecorder(stream, options);
        const recordedChunks = [];
        mediaRecorder.addEventListener('dataavailable', function(e) {
        if (e.data.size > 0) recordedChunks.push(e.data);
        });
        mediaRecorder.addEventListener('stop', function() {
            createDownloadLink(new Blob(recordedChunks));
          });
        stopButton.addEventListener('click', function() {
            console.log("stopButton clicked");
            //disable the stop button, enable the record too allow for new recordings 
            stopButton.disabled = true;
            recordButton.disabled = false;
            pauseButton.disabled = true;
            //reset button just in case the recording is stopped while paused 
            pauseButton.innerHTML = "Pause";
            mediaRecorder.stop();
        });

        mediaRecorder.start();
    };
    navigator.mediaDevices.getUserMedia({ audio: true, video: false }).then(handleSuccess);


}

function pauseRecording() {
    console.log("pauseButton clicked");
}
function createDownloadLink(blob) {
    var url = URL.createObjectURL(blob);
    var au = document.createElement('audio');
    var li = document.createElement('li');
    var link = document.createElement('a');
    //add controls to the <audio> element 
    au.controls = true;
    au.src = url;
    //link the a element to the blob 
    link.href = url;
    link.download = new Date().toISOString() + '.wav';
    link.innerHTML = link.download;
    //add the new audio and a elements to the li element 
    li.appendChild(au);
    li.appendChild(link);
    //add the li element to the ordered list 
    recordingsList.appendChild(li);
    var filename = new Date().toISOString();
    //filename to send to server without extension 
    //upload link 
    var upload = document.createElement('a');
    upload.href = "#";
    upload.innerHTML = "Upload";
    upload.addEventListener("click", function(event) {
        voice_text.disabled = false;
        var xhr = new XMLHttpRequest();
        xhr.onload = function(e) {
            if (this.readyState === 4) {
                console.log("Server returned: ", e.target.responseText);
            }
        };
        var fd = new FormData();
        fd.append("audio_data", blob, filename);
        xhr.open("POST", "upload", true);
        xhr.send(fd);
    })
    li.appendChild(document.createTextNode(" ")) //add a space in between 
    li.appendChild(upload) //add the upload link to li
}