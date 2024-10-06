document.addEventListener('DOMContentLoaded', () => {
    const addTradeForm = document.getElementById('addTradeForm');
    if (addTradeForm) {
        addTradeForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(addTradeForm);
            const response = await fetch('/trade', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            alert(result.message);
        });
    }
});