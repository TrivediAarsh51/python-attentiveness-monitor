from flask import Flask,render_template,request, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime



app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///todo1.db"
db = SQLAlchemy(app)

class Todo(db.Model):
    sno = db.Column(db.Integer,primary_key=True)
    title = db.Column(db.String(200),nullable=False)
    desc = db.Column(db.String(500),nullable=False)
    date_created = db.Column(db.DateTime,default=datetime.utcnow)

def __repr__ (self) -> str:
    return f"{self.sno} - {self.title}"
# db.init_app(app)


@app.route("/", methods=['GET','POST'])
def hello_world():
    if request.method == "POST":
        title=request.form['title']
        desc=request.form['desc']
        todo = Todo(title=title, desc=desc)
        db.session.add(todo)
        db.session.commit()

    alltodo = Todo.query.all()
    return render_template("index.html",alltodo=alltodo)

@app.route("/show")
def alltodos():
    alltodo = Todo.query.all()
    return "this is the products page"


@app.route("/update/int:sno")
def update(sno):
    todo = Todo.query.filter_by(sno=sno).first()
    
    db.session.commit()
    return redirect("/")


@app.route("/delete/<int:sno>")
def delete(sno):
    todo = Todo.query.filter_by(sno=sno).first()
    db.session.delete(todo)
    db.session.commit()
    return redirect("/")
    return "This element has been deleted"


with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)
