def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad())
    maxdiff = (df - t.grad()).abs().max().item()
    print(f"{s:15s} | exact: {str(ex):5s} | appr: {str(app):.5s} | maxdiff: {maxdiff}")
