let camera_button = document.querySelector("#start-camera");
let video = document.querySelector("#video");
let click_button = document.querySelector("#click-photo");
let canvas = document.querySelector("#canvas");

camera_button.addEventListener('click', async function() {
   	let stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
	video.srcObject = stream;
});

click_button.addEventListener('click', function() {
   	canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
   	let image_data_url = canvas.toDataURL('image/jpeg');

   	// data url of the image
   	console.log(image_data_url);
});


create table Que_8(empid int,salary int);
insert into Que_8 values(1,12340000);
insert into Que_8 values(2,12342000);
insert into Que_8 values(3,12345000);
insert into Que_8 values(4,12340000);
insert into Que_8 values(5,12348000);
insert into Que_8 values(6,12346000);
insert into Que_8 values(7,12343000);
insert into Que_8 values(8,12345000);
select * from Que_8;
select salary from Que_8 order by salary desc limit 2,1; 