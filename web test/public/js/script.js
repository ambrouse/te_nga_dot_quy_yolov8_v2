

document.addEventListener("DOMContentLoaded", function() {
    const screenWidth = window.screen.width;
    const screenHeight = window.screen.height;

    document.getElementsByClassName("container")[0].style.height = screenHeight.toString()+"px";

})