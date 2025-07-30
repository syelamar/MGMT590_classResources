
## Dataflow Pipeline

```
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import csv
import datetime

# Define pipeline options
options = PipelineOptions(
    runner='DataflowRunner',
    project='mgmt590-465220',
    region='us-central1',
    temp_location='gs://sameeryelamarthi-lab2/temp',
    staging_location='gs://sameeryelamarthi-lab2/staging'
)

def run():
    with beam.Pipeline(options=options) as pipeline:
        (
            pipeline
            | 'ReadFromGCS' >> beam.io.ReadFromText(
                'gs://sameeryelamarthi-lab2/NVidia_stock_history.csv',
                skip_header_lines=1
            )
            | 'ParseCSV' >> beam.Map(lambda line: next(csv.reader([line])))
            | 'TransformDate' >> beam.Map(lambda record: {
                'Date': datetime.datetime.fromisoformat(record[0]).date().isoformat(),
                'Open': float(record[1]),
                'High': float(record[2]),
                'Low': float(record[3]),
                'Close': float(record[4]),
                'Volume': int(record[5])
            })
            | 'WriteToBigQuery' >> beam.io.WriteToBigQuery(
                table='mgmt590-465220:lab_2.nvidia_stock',
                schema='Date:DATE,Open:FLOAT,High:FLOAT,Low:FLOAT,Close:FLOAT,Volume:INTEGER',
                write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED
            )
        )

if __name__ == '__main__':
    run()
```

## Sales Data Pipeline
```
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import csv
import datetime

options = PipelineOptions(
    runner='DataflowRunner',
    project='mgmt590-465220',
    region='us-central1',
    temp_location='gs://sameeryelamarthi_assignment_2/temp',
    staging_location='gs://sameeryelamarthi_assignment_2/staging'
)

def run():
    with beam.Pipeline(options=options) as pipeline:
        (
            pipeline
            | 'Read Sales CSV' >> beam.io.ReadFromText(
                'gs://sameeryelamarthi_assignment_2/kaggle-store-sales/train.csv',
                skip_header_lines=1
            )
            | 'Parse CSV' >> beam.Map(lambda line: next(csv.reader([line])))
            | 'To Dict' >> beam.Map(lambda record: {
                'id': int(record[0]),
                'date': datetime.datetime.fromisoformat(record[1]).date().isoformat(),
                'store_nbr': int(record[2]),
                'family': record[3],
                'sales': float(record[4]),
                'onpromotion': int(record[5])
            })
            | 'Write to BQ' >> beam.io.WriteToBigQuery(
                table='mgmt590-465220:assignment_2.sales_data',
                schema='id:INTEGER,date:DATE,store_nbr:INTEGER,family:STRING,sales:FLOAT,onpromotion:INTEGER',
                write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED
            )
        )

if __name__ == '__main__':
    run()
```

## Store Data Pipeline

```
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import csv

options = PipelineOptions(
    runner='DataflowRunner',
    project='mgmt590-465220',
    region='us-central1',
    temp_location='gs://sameeryelamarthi_assignment_2/temp',
    staging_location='gs://sameeryelamarthi_assignment_2/staging'
)

def run():
    with beam.Pipeline(options=options) as pipeline:
        (
            pipeline
            | 'Read Stores CSV' >> beam.io.ReadFromText(
                'gs://sameeryelamarthi_assignment_2/kaggle-store-sales/stores.csv',
                skip_header_lines=1
            )
            | 'Parse CSV' >> beam.Map(lambda line: next(csv.reader([line])))
            | 'To Dict' >> beam.Map(lambda record: {
                'store_nbr': int(record[0]),
                'city': record[1],
                'state': record[2],
                'type': record[3],
                'cluster': int(record[4])
            })
            | 'Write to BQ' >> beam.io.WriteToBigQuery(
                table='mgmt590-465220:assignment_2.store_info',
                schema='store_nbr:INTEGER,city:STRING,state:STRING,type:STRING,cluster:INTEGER',
                write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE,
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED
            )
        )

if __name__ == '__main__':
    run()
```
