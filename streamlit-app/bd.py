import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import sessionmaker


try:
    conn = psycopg2.connect(dbname='testdb', user='user', password='superpassword', host='localhost', port='5432')
    cursor = conn.cursor()
except:
    print('Can`t establish connection to database')

engine = create_engine("postgresql://user:superpassword@localhost/testdb")

Base = declarative_base()

class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)
    address = Column(String)

    def __repr__(self):
        return f"<Customer(id={self.id}, name={self.name}, email={self.email}, address={self.address})>"

Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

new_customer = Customer(name="Genry", email="genry@example.com", address="11111 Main Street")
session.add(new_customer)
session.commit()

session.close()