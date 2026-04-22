<?php 

session_start();
if (!isset($_SESSION['email'])) {
    header("Location: index.php");
    exit();
}

?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>User Page</title>
    <link rel="stylesheet" href="style.css">
</head>
<body style="background: #fff;">
    
    <div class = "box">
        <h1>Welcome <span><?= $_SESSION['name']; ?></span>! </h1>
        <p>Ready to find your next favourite movie?</p>
        <button onclick="window.location.href='http://127.0.0.1:5000?user=<?= urlencode($_SESSION['name']); ?>'">
            🎬 Get Recommendations
        </button>

        <button onclick="window.location.href='logout.php'">Logout</button>
    </div>

</body>
</html>
