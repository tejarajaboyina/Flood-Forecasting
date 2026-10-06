let graph;

function getPrediction() {
    let state = document.getElementById("state").value;
    let days = document.getElementById("days").value;

    fetch('/predict', {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ state, days })
    })
    .then(res => res.json())
    .then(data => {

        document.getElementById("risk").innerHTML =
            "Flood Risk: " + data.risk.toFixed(2) + "%";

        let labels = [];
        for (let i = 1; i <= data.predictions.length; i++) {
            labels.push("Day " + i);
        }

        if (graph) graph.destroy();

        const ctx = document.getElementById("graph").getContext("2d");
        graph = new Chart(ctx, {
            type: "line",
            data: {
                labels: labels,
                datasets: [{
                    label: "Predicted Water Level",
                    data: data.predictions,
                    borderWidth: 2
                }]
            }
        });
    });
}
