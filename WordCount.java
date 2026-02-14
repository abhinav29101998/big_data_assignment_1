import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class WordCount {

    // Mapper: converts each word into (word, 1)
    public static class WordMapper extends Mapper<LongWritable, Text, Text, IntWritable> {

        private static final IntWritable ONE_COUNT = new IntWritable(1);
        private Text outputWord = new Text();

        @Override
        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {

            // Remove punctuation, normalize case, then tokenize
            String cleanedText = value.toString()
                    .replaceAll("[^a-zA-Z0-9 ]", "")
                    .toLowerCase();

            StringTokenizer tokenizer = new StringTokenizer(cleanedText);

            while (tokenizer.hasMoreTokens()) {
                outputWord.set(tokenizer.nextToken());
                context.write(outputWord, ONE_COUNT);
            }
        }
    }

    // Reducer: aggregates counts for each word
    public static class WordReducer extends Reducer<Text, IntWritable, Text, IntWritable> {

        private IntWritable totalCount = new IntWritable();

        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {

            int countSum = 0;
            for (IntWritable num : values) {
                countSum += num.get();
            }

            totalCount.set(countSum);
            context.write(key, totalCount);
        }
    }

    public static void main(String[] args) throws Exception {

        Configuration configuration = new Configuration();
        Job job = Job.getInstance(configuration, "Word Frequency Counter");

        job.setJarByClass(WordCount.class);

        job.setMapperClass(WordMapper.class);
        job.setCombinerClass(WordReducer.class);
        job.setReducerClass(WordReducer.class);

        // Output key/value types
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        // Configure input split size (example: 1 MB)
        job.getConfiguration().setLong(
                "mapreduce.input.fileinputformat.split.maxsize",
                1024 * 1024
        );

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        // Track execution duration
        long begin = System.currentTimeMillis();
        boolean completed = job.waitForCompletion(true);
        long finish = System.currentTimeMillis();

        if (completed) {
            System.out.println("===========================================");
            System.out.println("Job completed in: " + (finish - begin) + " ms");
            System.out.println("===========================================");
            System.exit(0);
        } else {
            System.exit(1);
        }
    }
}
