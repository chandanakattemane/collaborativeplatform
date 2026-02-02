// preview in registration modal
document.addEventListener('DOMContentLoaded', function () {
  const regFile = document.getElementById('reg_profile_pic');
  const regPreview = document.getElementById('reg_preview');

  if (regFile) {
    regFile.addEventListener('change', function (e) {
      const f = e.target.files[0];
      if (!f) return;
      const reader = new FileReader();
      reader.onload = function (ev) { regPreview.src = ev.target.result; };
      reader.readAsDataURL(f);
    });
  }

  // edit-profile preview (if you want)
  const editFile = document.getElementById('edit_profile_pic');
  if (editFile) {
    editFile.addEventListener('change', function (e) {
      // optional: display a small preview - left as exercise
    });
  }
});
